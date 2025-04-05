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
_ApproxTopKOutput = collections.namedtuple(
    "ApproxTopK",
    ["values", "indices"])


TV_ApproxTopK_T = TypeVar("TV_ApproxTopK_T", _atypes.BFloat16, _atypes.Float32, _atypes.Half)

@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('approx_top_k')
def approx_top_k(input: Annotated[Any, TV_ApproxTopK_T], k: int, reduction_dimension:int=-1, recall_target:float=0.95, is_max_k:bool=True, reduction_input_size_override:int=-1, aggregate_to_topk:bool=True, name=None):
  r"""Returns min/max k values and their indices of the input operand in an approximate manner.

  See https://arxiv.org/abs/2206.14286 for the algorithm details.
  This op is only optimized on TPU currently.

  Args:
    input: A `Tensor`. Must be one of the following types: `half`, `bfloat16`, `float32`.
      Array to search. Must be at least 1-D of the floating type
    k: An `int` that is `>= 0`. Specifies the number of min/max-k.
    reduction_dimension: An optional `int`. Defaults to `-1`.
      Integer dimension along which to search. Default: -1.
    recall_target: An optional `float`. Defaults to `0.95`.
      Recall target for the approximation. Range in (0,1]
    is_max_k: An optional `bool`. Defaults to `True`.
      When true, computes max-k; otherwise computes min-k.
    reduction_input_size_override: An optional `int`. Defaults to `-1`.
      When set to a positive value, it overrides the size determined by
      `input[reduction_dim]` for evaluating the recall. This option is useful when
      the given `input` is only a subset of the overall computation in SPMD or
      distributed pipelines, where the true input size cannot be deferred by the
      `input` shape.
    aggregate_to_topk: An optional `bool`. Defaults to `True`.
      When true, aggregates approximate results to top-k. When false, returns the
      approximate results. The number of the approximate results is implementation
      defined and is greater equals to the specified `k`.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (values, indices).

    values: A `Tensor`. Has the same type as `input`.
    indices: A `Tensor` of type `int32`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "ApproxTopK", name, input, "k", k, "reduction_dimension",
        reduction_dimension, "recall_target", recall_target, "is_max_k",
        is_max_k, "reduction_input_size_override",
        reduction_input_size_override, "aggregate_to_topk", aggregate_to_topk)
      _result = _ApproxTopKOutput._make(_result)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_approx_top_k(
          (input, k, reduction_dimension, recall_target, is_max_k,
          reduction_input_size_override, aggregate_to_topk, name,), None)
      if _result is not NotImplemented:
        return _result
      return approx_top_k_eager_fallback(
          input, k=k, reduction_dimension=reduction_dimension,
          recall_target=recall_target, is_max_k=is_max_k,
          reduction_input_size_override=reduction_input_size_override,
          aggregate_to_topk=aggregate_to_topk, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            approx_top_k, (), dict(input=input, k=k,
                                   reduction_dimension=reduction_dimension,
                                   recall_target=recall_target,
                                   is_max_k=is_max_k,
                                   reduction_input_size_override=reduction_input_size_override,
                                   aggregate_to_topk=aggregate_to_topk,
                                   name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_approx_top_k(
        (input, k, reduction_dimension, recall_target, is_max_k,
        reduction_input_size_override, aggregate_to_topk, name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  k = _execute.make_int(k, "k")
  if reduction_dimension is None:
    reduction_dimension = -1
  reduction_dimension = _execute.make_int(reduction_dimension, "reduction_dimension")
  if recall_target is None:
    recall_target = 0.95
  recall_target = _execute.make_float(recall_target, "recall_target")
  if is_max_k is None:
    is_max_k = True
  is_max_k = _execute.make_bool(is_max_k, "is_max_k")
  if reduction_input_size_override is None:
    reduction_input_size_override = -1
  reduction_input_size_override = _execute.make_int(reduction_input_size_override, "reduction_input_size_override")
  if aggregate_to_topk is None:
    aggregate_to_topk = True
  aggregate_to_topk = _execute.make_bool(aggregate_to_topk, "aggregate_to_topk")
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "ApproxTopK", input=input, k=k,
                      reduction_dimension=reduction_dimension,
                      recall_target=recall_target, is_max_k=is_max_k,
                      reduction_input_size_override=reduction_input_size_override,
                      aggregate_to_topk=aggregate_to_topk, name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          approx_top_k, (), dict(input=input, k=k,
                                 reduction_dimension=reduction_dimension,
                                 recall_target=recall_target,
                                 is_max_k=is_max_k,
                                 reduction_input_size_override=reduction_input_size_override,
                                 aggregate_to_topk=aggregate_to_topk,
                                 name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("k", _op._get_attr_int("k"), "reduction_dimension",
              _op._get_attr_int("reduction_dimension"), "recall_target",
              _op.get_attr("recall_target"), "is_max_k",
              _op._get_attr_bool("is_max_k"), "reduction_input_size_override",
              _op._get_attr_int("reduction_input_size_override"),
              "aggregate_to_topk", _op._get_attr_bool("aggregate_to_topk"),
              "T", _op._get_attr_type("T"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "ApproxTopK", _inputs_flat, _attrs, _result)
  _result = _ApproxTopKOutput._make(_result)
  return _result

ApproxTopK = tf_export("raw_ops.ApproxTopK")(_ops.to_raw_op(approx_top_k))
_dispatcher_for_approx_top_k = approx_top_k._tf_type_based_dispatcher.Dispatch


def approx_top_k_eager_fallback(input: Annotated[Any, TV_ApproxTopK_T], k: int, reduction_dimension: int, recall_target: float, is_max_k: bool, reduction_input_size_override: int, aggregate_to_topk: bool, name, ctx):
  k = _execute.make_int(k, "k")
  if reduction_dimension is None:
    reduction_dimension = -1
  reduction_dimension = _execute.make_int(reduction_dimension, "reduction_dimension")
  if recall_target is None:
    recall_target = 0.95
  recall_target = _execute.make_float(recall_target, "recall_target")
  if is_max_k is None:
    is_max_k = True
  is_max_k = _execute.make_bool(is_max_k, "is_max_k")
  if reduction_input_size_override is None:
    reduction_input_size_override = -1
  reduction_input_size_override = _execute.make_int(reduction_input_size_override, "reduction_input_size_override")
  if aggregate_to_topk is None:
    aggregate_to_topk = True
  aggregate_to_topk = _execute.make_bool(aggregate_to_topk, "aggregate_to_topk")
  _attr_T, (input,) = _execute.args_to_matching_eager([input], ctx, [_dtypes.half, _dtypes.bfloat16, _dtypes.float32, ])
  _inputs_flat = [input]
  _attrs = ("k", k, "reduction_dimension", reduction_dimension,
  "recall_target", recall_target, "is_max_k", is_max_k,
  "reduction_input_size_override", reduction_input_size_override,
  "aggregate_to_topk", aggregate_to_topk, "T", _attr_T)
  _result = _execute.execute(b"ApproxTopK", 2, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "ApproxTopK", _inputs_flat, _attrs, _result)
  _result = _ApproxTopKOutput._make(_result)
  return _result


TV_AvgPool_T = TypeVar("TV_AvgPool_T", _atypes.BFloat16, _atypes.Float32, _atypes.Float64, _atypes.Half)

def avg_pool(value: Annotated[Any, TV_AvgPool_T], ksize, strides, padding: str, data_format:str="NHWC", name=None) -> Annotated[Any, TV_AvgPool_T]:
  r"""Performs average pooling on the input.

  Each entry in `output` is the mean of the corresponding size `ksize`
  window in `value`.

  Args:
    value: A `Tensor`. Must be one of the following types: `half`, `bfloat16`, `float32`, `float64`.
      4-D with shape `[batch, height, width, channels]`.
    ksize: A list of `ints` that has length `>= 4`.
      The size of the sliding window for each dimension of `value`.
    strides: A list of `ints` that has length `>= 4`.
      The stride of the sliding window for each dimension of `value`.
    padding: A `string` from: `"SAME", "VALID"`.
      The type of padding algorithm to use.
    data_format: An optional `string` from: `"NHWC", "NCHW"`. Defaults to `"NHWC"`.
      Specify the data format of the input and output data. With the
      default format "NHWC", the data is stored in the order of:
          [batch, in_height, in_width, in_channels].
      Alternatively, the format could be "NCHW", the data storage order of:
          [batch, in_channels, in_height, in_width].
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `value`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "AvgPool", name, value, "ksize", ksize, "strides", strides,
        "padding", padding, "data_format", data_format)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return avg_pool_eager_fallback(
          value, ksize=ksize, strides=strides, padding=padding,
          data_format=data_format, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if not isinstance(ksize, (list, tuple)):
    raise TypeError(
        "Expected list for 'ksize' argument to "
        "'avg_pool' Op, not %r." % ksize)
  ksize = [_execute.make_int(_i, "ksize") for _i in ksize]
  if not isinstance(strides, (list, tuple)):
    raise TypeError(
        "Expected list for 'strides' argument to "
        "'avg_pool' Op, not %r." % strides)
  strides = [_execute.make_int(_i, "strides") for _i in strides]
  padding = _execute.make_str(padding, "padding")
  if data_format is None:
    data_format = "NHWC"
  data_format = _execute.make_str(data_format, "data_format")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "AvgPool", value=value, ksize=ksize, strides=strides, padding=padding,
                   data_format=data_format, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("ksize", _op.get_attr("ksize"), "strides",
              _op.get_attr("strides"), "padding", _op.get_attr("padding"),
              "data_format", _op.get_attr("data_format"), "T",
              _op._get_attr_type("T"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "AvgPool", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

AvgPool = tf_export("raw_ops.AvgPool")(_ops.to_raw_op(avg_pool))


def avg_pool_eager_fallback(value: Annotated[Any, TV_AvgPool_T], ksize, strides, padding: str, data_format: str, name, ctx) -> Annotated[Any, TV_AvgPool_T]:
  if not isinstance(ksize, (list, tuple)):
    raise TypeError(
        "Expected list for 'ksize' argument to "
        "'avg_pool' Op, not %r." % ksize)
  ksize = [_execute.make_int(_i, "ksize") for _i in ksize]
  if not isinstance(strides, (list, tuple)):
    raise TypeError(
        "Expected list for 'strides' argument to "
        "'avg_pool' Op, not %r." % strides)
  strides = [_execute.make_int(_i, "strides") for _i in strides]
  padding = _execute.make_str(padding, "padding")
  if data_format is None:
    data_format = "NHWC"
  data_format = _execute.make_str(data_format, "data_format")
  _attr_T, (value,) = _execute.args_to_matching_eager([value], ctx, [_dtypes.half, _dtypes.bfloat16, _dtypes.float32, _dtypes.float64, ])
  _inputs_flat = [value]
  _attrs = ("ksize", ksize, "strides", strides, "padding", padding,
  "data_format", data_format, "T", _attr_T)
  _result = _execute.execute(b"AvgPool", 1, inputs=_inputs_flat, attrs=_attrs,
                             ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "AvgPool", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_AvgPool3D_T = TypeVar("TV_AvgPool3D_T", _atypes.BFloat16, _atypes.Float32, _atypes.Float64, _atypes.Half)

def avg_pool3d(input: Annotated[Any, TV_AvgPool3D_T], ksize, strides, padding: str, data_format:str="NDHWC", name=None) -> Annotated[Any, TV_AvgPool3D_T]:
  r"""Performs 3D average pooling on the input.

  Each entry in `output` is the mean of the corresponding size `ksize` window in
  `value`.

  Args:
    input: A `Tensor`. Must be one of the following types: `half`, `bfloat16`, `float32`, `float64`.
      Shape `[batch, depth, rows, cols, channels]` tensor to pool over.
    ksize: A list of `ints` that has length `>= 5`.
      1-D tensor of length 5. The size of the window for each dimension of
      the input tensor. Must have `ksize[0] = ksize[4] = 1`.
    strides: A list of `ints` that has length `>= 5`.
      1-D tensor of length 5. The stride of the sliding window for each
      dimension of `input`. Must have `strides[0] = strides[4] = 1`.
    padding: A `string` from: `"SAME", "VALID"`.
      The type of padding algorithm to use.
    data_format: An optional `string` from: `"NDHWC", "NCDHW"`. Defaults to `"NDHWC"`.
      The data format of the input and output data. With the
      default format "NDHWC", the data is stored in the order of:
          [batch, in_depth, in_height, in_width, in_channels].
      Alternatively, the format could be "NCDHW", the data storage order is:
          [batch, in_channels, in_depth, in_height, in_width].
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "AvgPool3D", name, input, "ksize", ksize, "strides", strides,
        "padding", padding, "data_format", data_format)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return avg_pool3d_eager_fallback(
          input, ksize=ksize, strides=strides, padding=padding,
          data_format=data_format, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if not isinstance(ksize, (list, tuple)):
    raise TypeError(
        "Expected list for 'ksize' argument to "
        "'avg_pool3d' Op, not %r." % ksize)
  ksize = [_execute.make_int(_i, "ksize") for _i in ksize]
  if not isinstance(strides, (list, tuple)):
    raise TypeError(
        "Expected list for 'strides' argument to "
        "'avg_pool3d' Op, not %r." % strides)
  strides = [_execute.make_int(_i, "strides") for _i in strides]
  padding = _execute.make_str(padding, "padding")
  if data_format is None:
    data_format = "NDHWC"
  data_format = _execute.make_str(data_format, "data_format")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "AvgPool3D", input=input, ksize=ksize, strides=strides,
                     padding=padding, data_format=data_format, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("ksize", _op.get_attr("ksize"), "strides",
              _op.get_attr("strides"), "padding", _op.get_attr("padding"),
              "data_format", _op.get_attr("data_format"), "T",
              _op._get_attr_type("T"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "AvgPool3D", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

AvgPool3D = tf_export("raw_ops.AvgPool3D")(_ops.to_raw_op(avg_pool3d))


def avg_pool3d_eager_fallback(input: Annotated[Any, TV_AvgPool3D_T], ksize, strides, padding: str, data_format: str, name, ctx) -> Annotated[Any, TV_AvgPool3D_T]:
  if not isinstance(ksize, (list, tuple)):
    raise TypeError(
        "Expected list for 'ksize' argument to "
        "'avg_pool3d' Op, not %r." % ksize)
  ksize = [_execute.make_int(_i, "ksize") for _i in ksize]
  if not isinstance(strides, (list, tuple)):
    raise TypeError(
        "Expected list for 'strides' argument to "
        "'avg_pool3d' Op, not %r." % strides)
  strides = [_execute.make_int(_i, "strides") for _i in strides]
  padding = _execute.make_str(padding, "padding")
  if data_format is None:
    data_format = "NDHWC"
  data_format = _execute.make_str(data_format, "data_format")
  _attr_T, (input,) = _execute.args_to_matching_eager([input], ctx, [_dtypes.half, _dtypes.bfloat16, _dtypes.float32, _dtypes.float64, ])
  _inputs_flat = [input]
  _attrs = ("ksize", ksize, "strides", strides, "padding", padding,
  "data_format", data_format, "T", _attr_T)
  _result = _execute.execute(b"AvgPool3D", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "AvgPool3D", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_AvgPool3DGrad_T = TypeVar("TV_AvgPool3DGrad_T", _atypes.BFloat16, _atypes.Float32, _atypes.Float64, _atypes.Half)

def avg_pool3d_grad(orig_input_shape: Annotated[Any, _atypes.Int32], grad: Annotated[Any, TV_AvgPool3DGrad_T], ksize, strides, padding: str, data_format:str="NDHWC", name=None) -> Annotated[Any, TV_AvgPool3DGrad_T]:
  r"""Computes gradients of average pooling function.

  Args:
    orig_input_shape: A `Tensor` of type `int32`.
      The original input dimensions.
    grad: A `Tensor`. Must be one of the following types: `half`, `bfloat16`, `float32`, `float64`.
      Output backprop of shape `[batch, depth, rows, cols, channels]`.
    ksize: A list of `ints` that has length `>= 5`.
      1-D tensor of length 5. The size of the window for each dimension of
      the input tensor. Must have `ksize[0] = ksize[4] = 1`.
    strides: A list of `ints` that has length `>= 5`.
      1-D tensor of length 5. The stride of the sliding window for each
      dimension of `input`. Must have `strides[0] = strides[4] = 1`.
    padding: A `string` from: `"SAME", "VALID"`.
      The type of padding algorithm to use.
    data_format: An optional `string` from: `"NDHWC", "NCDHW"`. Defaults to `"NDHWC"`.
      The data format of the input and output data. With the
      default format "NDHWC", the data is stored in the order of:
          [batch, in_depth, in_height, in_width, in_channels].
      Alternatively, the format could be "NCDHW", the data storage order is:
          [batch, in_channels, in_depth, in_height, in_width].
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `grad`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "AvgPool3DGrad", name, orig_input_shape, grad, "ksize", ksize,
        "strides", strides, "padding", padding, "data_format", data_format)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return avg_pool3d_grad_eager_fallback(
          orig_input_shape, grad, ksize=ksize, strides=strides,
          padding=padding, data_format=data_format, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if not isinstance(ksize, (list, tuple)):
    raise TypeError(
        "Expected list for 'ksize' argument to "
        "'avg_pool3d_grad' Op, not %r." % ksize)
  ksize = [_execute.make_int(_i, "ksize") for _i in ksize]
  if not isinstance(strides, (list, tuple)):
    raise TypeError(
        "Expected list for 'strides' argument to "
        "'avg_pool3d_grad' Op, not %r." % strides)
  strides = [_execute.make_int(_i, "strides") for _i in strides]
  padding = _execute.make_str(padding, "padding")
  if data_format is None:
    data_format = "NDHWC"
  data_format = _execute.make_str(data_format, "data_format")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "AvgPool3DGrad", orig_input_shape=orig_input_shape, grad=grad,
                         ksize=ksize, strides=strides, padding=padding,
                         data_format=data_format, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("ksize", _op.get_attr("ksize"), "strides",
              _op.get_attr("strides"), "padding", _op.get_attr("padding"),
              "data_format", _op.get_attr("data_format"), "T",
              _op._get_attr_type("T"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "AvgPool3DGrad", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

AvgPool3DGrad = tf_export("raw_ops.AvgPool3DGrad")(_ops.to_raw_op(avg_pool3d_grad))


def avg_pool3d_grad_eager_fallback(orig_input_shape: Annotated[Any, _atypes.Int32], grad: Annotated[Any, TV_AvgPool3DGrad_T], ksize, strides, padding: str, data_format: str, name, ctx) -> Annotated[Any, TV_AvgPool3DGrad_T]:
  if not isinstance(ksize, (list, tuple)):
    raise TypeError(
        "Expected list for 'ksize' argument to "
        "'avg_pool3d_grad' Op, not %r." % ksize)
  ksize = [_execute.make_int(_i, "ksize") for _i in ksize]
  if not isinstance(strides, (list, tuple)):
    raise TypeError(
        "Expected list for 'strides' argument to "
        "'avg_pool3d_grad' Op, not %r." % strides)
  strides = [_execute.make_int(_i, "strides") for _i in strides]
  padding = _execute.make_str(padding, "padding")
  if data_format is None:
    data_format = "NDHWC"
  data_format = _execute.make_str(data_format, "data_format")
  _attr_T, (grad,) = _execute.args_to_matching_eager([grad], ctx, [_dtypes.half, _dtypes.bfloat16, _dtypes.float32, _dtypes.float64, ])
  orig_input_shape = _ops.convert_to_tensor(orig_input_shape, _dtypes.int32)
  _inputs_flat = [orig_input_shape, grad]
  _attrs = ("ksize", ksize, "strides", strides, "padding", padding,
  "data_format", data_format, "T", _attr_T)
  _result = _execute.execute(b"AvgPool3DGrad", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "AvgPool3DGrad", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_AvgPoolGrad_T = TypeVar("TV_AvgPoolGrad_T", _atypes.BFloat16, _atypes.Float32, _atypes.Float64, _atypes.Half)

def avg_pool_grad(orig_input_shape: Annotated[Any, _atypes.Int32], grad: Annotated[Any, TV_AvgPoolGrad_T], ksize, strides, padding: str, data_format:str="NHWC", name=None) -> Annotated[Any, TV_AvgPoolGrad_T]:
  r"""Computes gradients of the average pooling function.

  Args:
    orig_input_shape: A `Tensor` of type `int32`.
      1-D.  Shape of the original input to `avg_pool`.
    grad: A `Tensor`. Must be one of the following types: `half`, `bfloat16`, `float32`, `float64`.
      4-D with shape `[batch, height, width, channels]`.  Gradients w.r.t.
      the output of `avg_pool`.
    ksize: A list of `ints` that has length `>= 4`.
      The size of the sliding window for each dimension of the input.
    strides: A list of `ints` that has length `>= 4`.
      The stride of the sliding window for each dimension of the input.
    padding: A `string` from: `"SAME", "VALID"`.
      The type of padding algorithm to use.
    data_format: An optional `string` from: `"NHWC", "NCHW"`. Defaults to `"NHWC"`.
      Specify the data format of the input and output data. With the
      default format "NHWC", the data is stored in the order of:
          [batch, in_height, in_width, in_channels].
      Alternatively, the format could be "NCHW", the data storage order of:
          [batch, in_channels, in_height, in_width].
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `grad`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "AvgPoolGrad", name, orig_input_shape, grad, "ksize", ksize,
        "strides", strides, "padding", padding, "data_format", data_format)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return avg_pool_grad_eager_fallback(
          orig_input_shape, grad, ksize=ksize, strides=strides,
          padding=padding, data_format=data_format, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if not isinstance(ksize, (list, tuple)):
    raise TypeError(
        "Expected list for 'ksize' argument to "
        "'avg_pool_grad' Op, not %r." % ksize)
  ksize = [_execute.make_int(_i, "ksize") for _i in ksize]
  if not isinstance(strides, (list, tuple)):
    raise TypeError(
        "Expected list for 'strides' argument to "
        "'avg_pool_grad' Op, not %r." % strides)
  strides = [_execute.make_int(_i, "strides") for _i in strides]
  padding = _execute.make_str(padding, "padding")
  if data_format is None:
    data_format = "NHWC"
  data_format = _execute.make_str(data_format, "data_format")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "AvgPoolGrad", orig_input_shape=orig_input_shape, grad=grad,
                       ksize=ksize, strides=strides, padding=padding,
                       data_format=data_format, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("ksize", _op.get_attr("ksize"), "strides",
              _op.get_attr("strides"), "padding", _op.get_attr("padding"),
              "data_format", _op.get_attr("data_format"), "T",
              _op._get_attr_type("T"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "AvgPoolGrad", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

AvgPoolGrad = tf_export("raw_ops.AvgPoolGrad")(_ops.to_raw_op(avg_pool_grad))


def avg_pool_grad_eager_fallback(orig_input_shape: Annotated[Any, _atypes.Int32], grad: Annotated[Any, TV_AvgPoolGrad_T], ksize, strides, padding: str, data_format: str, name, ctx) -> Annotated[Any, TV_AvgPoolGrad_T]:
  if not isinstance(ksize, (list, tuple)):
    raise TypeError(
        "Expected list for 'ksize' argument to "
        "'avg_pool_grad' Op, not %r." % ksize)
  ksize = [_execute.make_int(_i, "ksize") for _i in ksize]
  if not isinstance(strides, (list, tuple)):
    raise TypeError(
        "Expected list for 'strides' argument to "
        "'avg_pool_grad' Op, not %r." % strides)
  strides = [_execute.make_int(_i, "strides") for _i in strides]
  padding = _execute.make_str(padding, "padding")
  if data_format is None:
    data_format = "NHWC"
  data_format = _execute.make_str(data_format, "data_format")
  _attr_T, (grad,) = _execute.args_to_matching_eager([grad], ctx, [_dtypes.half, _dtypes.bfloat16, _dtypes.float32, _dtypes.float64, ])
  orig_input_shape = _ops.convert_to_tensor(orig_input_shape, _dtypes.int32)
  _inputs_flat = [orig_input_shape, grad]
  _attrs = ("ksize", ksize, "strides", strides, "padding", padding,
  "data_format", data_format, "T", _attr_T)
  _result = _execute.execute(b"AvgPoolGrad", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "AvgPoolGrad", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_BatchNormWithGlobalNormalization_T = TypeVar("TV_BatchNormWithGlobalNormalization_T", _atypes.BFloat16, _atypes.Complex128, _atypes.Complex64, _atypes.Float32, _atypes.Float64, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.UInt16, _atypes.UInt32, _atypes.UInt64, _atypes.UInt8)

def _batch_norm_with_global_normalization(t: Annotated[Any, TV_BatchNormWithGlobalNormalization_T], m: Annotated[Any, TV_BatchNormWithGlobalNormalization_T], v: Annotated[Any, TV_BatchNormWithGlobalNormalization_T], beta: Annotated[Any, TV_BatchNormWithGlobalNormalization_T], gamma: Annotated[Any, TV_BatchNormWithGlobalNormalization_T], variance_epsilon: float, scale_after_normalization: bool, name=None) -> Annotated[Any, TV_BatchNormWithGlobalNormalization_T]:
  r"""Batch normalization.

  This op is deprecated. Prefer `tf.nn.batch_normalization`.

  Args:
    t: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`, `complex64`, `int64`, `qint8`, `quint8`, `qint32`, `bfloat16`, `qint16`, `quint16`, `uint16`, `complex128`, `half`, `uint32`, `uint64`.
      A 4D input Tensor.
    m: A `Tensor`. Must have the same type as `t`.
      A 1D mean Tensor with size matching the last dimension of t.
      This is the first output from tf.nn.moments,
      or a saved moving average thereof.
    v: A `Tensor`. Must have the same type as `t`.
      A 1D variance Tensor with size matching the last dimension of t.
      This is the second output from tf.nn.moments,
      or a saved moving average thereof.
    beta: A `Tensor`. Must have the same type as `t`.
      A 1D beta Tensor with size matching the last dimension of t.
      An offset to be added to the normalized tensor.
    gamma: A `Tensor`. Must have the same type as `t`.
      A 1D gamma Tensor with size matching the last dimension of t.
      If "scale_after_normalization" is true, this tensor will be multiplied
      with the normalized tensor.
    variance_epsilon: A `float`. A small float number to avoid dividing by 0.
    scale_after_normalization: A `bool`.
      A bool indicating whether the resulted tensor
      needs to be multiplied with gamma.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `t`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "BatchNormWithGlobalNormalization", name, t, m, v, beta, gamma,
        "variance_epsilon", variance_epsilon, "scale_after_normalization",
        scale_after_normalization)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return _batch_norm_with_global_normalization_eager_fallback(
          t, m, v, beta, gamma, variance_epsilon=variance_epsilon,
          scale_after_normalization=scale_after_normalization, name=name,
          ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  variance_epsilon = _execute.make_float(variance_epsilon, "variance_epsilon")
  scale_after_normalization = _execute.make_bool(scale_after_normalization, "scale_after_normalization")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "BatchNormWithGlobalNormalization", t=t, m=m, v=v, beta=beta,
                                            gamma=gamma,
                                            variance_epsilon=variance_epsilon,
                                            scale_after_normalization=scale_after_normalization,
                                            name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"), "variance_epsilon",
              _op.get_attr("variance_epsilon"), "scale_after_normalization",
              _op._get_attr_bool("scale_after_normalization"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "BatchNormWithGlobalNormalization", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

BatchNormWithGlobalNormalization = tf_export("raw_ops.BatchNormWithGlobalNormalization")(_ops.to_raw_op(_batch_norm_with_global_normalization))


def _batch_norm_with_global_normalization_eager_fallback(t: Annotated[Any, TV_BatchNormWithGlobalNormalization_T], m: Annotated[Any, TV_BatchNormWithGlobalNormalization_T], v: Annotated[Any, TV_BatchNormWithGlobalNormalization_T], beta: Annotated[Any, TV_BatchNormWithGlobalNormalization_T], gamma: Annotated[Any, TV_BatchNormWithGlobalNormalization_T], variance_epsilon: float, scale_after_normalization: bool, name, ctx) -> Annotated[Any, TV_BatchNormWithGlobalNormalization_T]:
  variance_epsilon = _execute.make_float(variance_epsilon, "variance_epsilon")
  scale_after_normalization = _execute.make_bool(scale_after_normalization, "scale_after_normalization")
  _attr_T, _inputs_T = _execute.args_to_matching_eager([t, m, v, beta, gamma], ctx, [_dtypes.float32, _dtypes.float64, _dtypes.int32, _dtypes.uint8, _dtypes.int16, _dtypes.int8, _dtypes.complex64, _dtypes.int64, _dtypes.qint8, _dtypes.quint8, _dtypes.qint32, _dtypes.bfloat16, _dtypes.qint16, _dtypes.quint16, _dtypes.uint16, _dtypes.complex128, _dtypes.half, _dtypes.uint32, _dtypes.uint64, ])
  (t, m, v, beta, gamma) = _inputs_T
  _inputs_flat = [t, m, v, beta, gamma]
  _attrs = ("T", _attr_T, "variance_epsilon", variance_epsilon,
  "scale_after_normalization", scale_after_normalization)
  _result = _execute.execute(b"BatchNormWithGlobalNormalization", 1,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "BatchNormWithGlobalNormalization", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

_BatchNormWithGlobalNormalizationGradOutput = collections.namedtuple(
    "BatchNormWithGlobalNormalizationGrad",
    ["dx", "dm", "dv", "db", "dg"])


TV_BatchNormWithGlobalNormalizationGrad_T = TypeVar("TV_BatchNormWithGlobalNormalizationGrad_T", _atypes.BFloat16, _atypes.Complex128, _atypes.Complex64, _atypes.Float32, _atypes.Float64, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.UInt16, _atypes.UInt32, _atypes.UInt64, _atypes.UInt8)

def batch_norm_with_global_normalization_grad(t: Annotated[Any, TV_BatchNormWithGlobalNormalizationGrad_T], m: Annotated[Any, TV_BatchNormWithGlobalNormalizationGrad_T], v: Annotated[Any, TV_BatchNormWithGlobalNormalizationGrad_T], gamma: Annotated[Any, TV_BatchNormWithGlobalNormalizationGrad_T], backprop: Annotated[Any, TV_BatchNormWithGlobalNormalizationGrad_T], variance_epsilon: float, scale_after_normalization: bool, name=None):
  r"""Gradients for batch normalization.

  This op is deprecated. See `tf.nn.batch_normalization`.

  Args:
    t: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`, `complex64`, `int64`, `qint8`, `quint8`, `qint32`, `bfloat16`, `qint16`, `quint16`, `uint16`, `complex128`, `half`, `uint32`, `uint64`.
      A 4D input Tensor.
    m: A `Tensor`. Must have the same type as `t`.
      A 1D mean Tensor with size matching the last dimension of t.
      This is the first output from tf.nn.moments,
      or a saved moving average thereof.
    v: A `Tensor`. Must have the same type as `t`.
      A 1D variance Tensor with size matching the last dimension of t.
      This is the second output from tf.nn.moments,
      or a saved moving average thereof.
    gamma: A `Tensor`. Must have the same type as `t`.
      A 1D gamma Tensor with size matching the last dimension of t.
      If "scale_after_normalization" is true, this Tensor will be multiplied
      with the normalized Tensor.
    backprop: A `Tensor`. Must have the same type as `t`. 4D backprop Tensor.
    variance_epsilon: A `float`. A small float number to avoid dividing by 0.
    scale_after_normalization: A `bool`.
      A bool indicating whether the resulted tensor
      needs to be multiplied with gamma.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (dx, dm, dv, db, dg).

    dx: A `Tensor`. Has the same type as `t`.
    dm: A `Tensor`. Has the same type as `t`.
    dv: A `Tensor`. Has the same type as `t`.
    db: A `Tensor`. Has the same type as `t`.
    dg: A `Tensor`. Has the same type as `t`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "BatchNormWithGlobalNormalizationGrad", name, t, m, v, gamma,
        backprop, "variance_epsilon", variance_epsilon,
        "scale_after_normalization", scale_after_normalization)
      _result = _BatchNormWithGlobalNormalizationGradOutput._make(_result)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return batch_norm_with_global_normalization_grad_eager_fallback(
          t, m, v, gamma, backprop, variance_epsilon=variance_epsilon,
          scale_after_normalization=scale_after_normalization, name=name,
          ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  variance_epsilon = _execute.make_float(variance_epsilon, "variance_epsilon")
  scale_after_normalization = _execute.make_bool(scale_after_normalization, "scale_after_normalization")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "BatchNormWithGlobalNormalizationGrad", t=t, m=m, v=v, gamma=gamma,
                                                backprop=backprop,
                                                variance_epsilon=variance_epsilon,
                                                scale_after_normalization=scale_after_normalization,
                                                name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"), "variance_epsilon",
              _op.get_attr("variance_epsilon"), "scale_after_normalization",
              _op._get_attr_bool("scale_after_normalization"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "BatchNormWithGlobalNormalizationGrad", _inputs_flat, _attrs, _result)
  _result = _BatchNormWithGlobalNormalizationGradOutput._make(_result)
  return _result

BatchNormWithGlobalNormalizationGrad = tf_export("raw_ops.BatchNormWithGlobalNormalizationGrad")(_ops.to_raw_op(batch_norm_with_global_normalization_grad))


def batch_norm_with_global_normalization_grad_eager_fallback(t: Annotated[Any, TV_BatchNormWithGlobalNormalizationGrad_T], m: Annotated[Any, TV_BatchNormWithGlobalNormalizationGrad_T], v: Annotated[Any, TV_BatchNormWithGlobalNormalizationGrad_T], gamma: Annotated[Any, TV_BatchNormWithGlobalNormalizationGrad_T], backprop: Annotated[Any, TV_BatchNormWithGlobalNormalizationGrad_T], variance_epsilon: float, scale_after_normalization: bool, name, ctx):
  variance_epsilon = _execute.make_float(variance_epsilon, "variance_epsilon")
  scale_after_normalization = _execute.make_bool(scale_after_normalization, "scale_after_normalization")
  _attr_T, _inputs_T = _execute.args_to_matching_eager([t, m, v, gamma, backprop], ctx, [_dtypes.float32, _dtypes.float64, _dtypes.int32, _dtypes.uint8, _dtypes.int16, _dtypes.int8, _dtypes.complex64, _dtypes.int64, _dtypes.qint8, _dtypes.quint8, _dtypes.qint32, _dtypes.bfloat16, _dtypes.qint16, _dtypes.quint16, _dtypes.uint16, _dtypes.complex128, _dtypes.half, _dtypes.uint32, _dtypes.uint64, ])
  (t, m, v, gamma, backprop) = _inputs_T
  _inputs_flat = [t, m, v, gamma, backprop]
  _attrs = ("T", _attr_T, "variance_epsilon", variance_epsilon,
  "scale_after_normalization", scale_after_normalization)
  _result = _execute.execute(b"BatchNormWithGlobalNormalizationGrad", 5,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "BatchNormWithGlobalNormalizationGrad", _inputs_flat, _attrs, _result)
  _result = _BatchNormWithGlobalNormalizationGradOutput._make(_result)
  return _result


TV_BiasAdd_T = TypeVar("TV_BiasAdd_T", _atypes.BFloat16, _atypes.Complex128, _atypes.Complex64, _atypes.Float32, _atypes.Float64, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.UInt16, _atypes.UInt32, _atypes.UInt64, _atypes.UInt8)

def bias_add(value: Annotated[Any, TV_BiasAdd_T], bias: Annotated[Any, TV_BiasAdd_T], data_format:str="NHWC", name=None) -> Annotated[Any, TV_BiasAdd_T]:
  r"""Adds `bias` to `value`.

  This is a special case of `tf.add` where `bias` is restricted to be 1-D.
  Broadcasting is supported, so `value` may have any number of dimensions.

  Args:
    value: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`, `complex64`, `int64`, `qint8`, `quint8`, `qint32`, `bfloat16`, `qint16`, `quint16`, `uint16`, `complex128`, `half`, `uint32`, `uint64`.
      Any number of dimensions.
    bias: A `Tensor`. Must have the same type as `value`.
      1-D with size the last dimension of `value`.
    data_format: An optional `string` from: `"NHWC", "NCHW"`. Defaults to `"NHWC"`.
      Specify the data format of the input and output data. With the
      default format "NHWC", the bias tensor will be added to the last dimension
      of the value tensor.
      Alternatively, the format could be "NCHW", the data storage order of:
          [batch, in_channels, in_height, in_width].
      The tensor will be added to "in_channels", the third-to-the-last
          dimension.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `value`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "BiasAdd", name, value, bias, "data_format", data_format)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return bias_add_eager_fallback(
          value, bias, data_format=data_format, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if data_format is None:
    data_format = "NHWC"
  data_format = _execute.make_str(data_format, "data_format")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "BiasAdd", value=value, bias=bias, data_format=data_format, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"), "data_format",
              _op.get_attr("data_format"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "BiasAdd", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

BiasAdd = tf_export("raw_ops.BiasAdd")(_ops.to_raw_op(bias_add))


def bias_add_eager_fallback(value: Annotated[Any, TV_BiasAdd_T], bias: Annotated[Any, TV_BiasAdd_T], data_format: str, name, ctx) -> Annotated[Any, TV_BiasAdd_T]:
  if data_format is None:
    data_format = "NHWC"
  data_format = _execute.make_str(data_format, "data_format")
  _attr_T, _inputs_T = _execute.args_to_matching_eager([value, bias], ctx, [_dtypes.float32, _dtypes.float64, _dtypes.int32, _dtypes.uint8, _dtypes.int16, _dtypes.int8, _dtypes.complex64, _dtypes.int64, _dtypes.qint8, _dtypes.quint8, _dtypes.qint32, _dtypes.bfloat16, _dtypes.qint16, _dtypes.quint16, _dtypes.uint16, _dtypes.complex128, _dtypes.half, _dtypes.uint32, _dtypes.uint64, ])
  (value, bias) = _inputs_T
  _inputs_flat = [value, bias]
  _attrs = ("T", _attr_T, "data_format", data_format)
  _result = _execute.execute(b"BiasAdd", 1, inputs=_inputs_flat, attrs=_attrs,
                             ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "BiasAdd", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_BiasAddGrad_T = TypeVar("TV_BiasAddGrad_T", _atypes.BFloat16, _atypes.Complex128, _atypes.Complex64, _atypes.Float32, _atypes.Float64, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.UInt16, _atypes.UInt32, _atypes.UInt64, _atypes.UInt8)

def bias_add_grad(out_backprop: Annotated[Any, TV_BiasAddGrad_T], data_format:str="NHWC", name=None) -> Annotated[Any, TV_BiasAddGrad_T]:
  r"""The backward operation for "BiasAdd" on the "bias" tensor.

  It accumulates all the values from out_backprop into the feature dimension.
  For NHWC data format, the feature dimension is the last. For NCHW data format,
  the feature dimension is the third-to-last.

  Args:
    out_backprop: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`, `complex64`, `int64`, `qint8`, `quint8`, `qint32`, `bfloat16`, `qint16`, `quint16`, `uint16`, `complex128`, `half`, `uint32`, `uint64`.
      Any number of dimensions.
    data_format: An optional `string` from: `"NHWC", "NCHW"`. Defaults to `"NHWC"`.
      Specify the data format of the input and output data. With the
      default format "NHWC", the bias tensor will be added to the last dimension
      of the value tensor.
      Alternatively, the format could be "NCHW", the data storage order of:
          [batch, in_channels, in_height, in_width].
      The tensor will be added to "in_channels", the third-to-the-last
          dimension.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `out_backprop`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "BiasAddGrad", name, out_backprop, "data_format", data_format)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return bias_add_grad_eager_fallback(
          out_backprop, data_format=data_format, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if data_format is None:
    data_format = "NHWC"
  data_format = _execute.make_str(data_format, "data_format")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "BiasAddGrad", out_backprop=out_backprop, data_format=data_format,
                       name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"), "data_format",
              _op.get_attr("data_format"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "BiasAddGrad", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

BiasAddGrad = tf_export("raw_ops.BiasAddGrad")(_ops.to_raw_op(bias_add_grad))


def bias_add_grad_eager_fallback(out_backprop: Annotated[Any, TV_BiasAddGrad_T], data_format: str, name, ctx) -> Annotated[Any, TV_BiasAddGrad_T]:
  if data_format is None:
    data_format = "NHWC"
  data_format = _execute.make_str(data_format, "data_format")
  _attr_T, (out_backprop,) = _execute.args_to_matching_eager([out_backprop], ctx, [_dtypes.float32, _dtypes.float64, _dtypes.int32, _dtypes.uint8, _dtypes.int16, _dtypes.int8, _dtypes.complex64, _dtypes.int64, _dtypes.qint8, _dtypes.quint8, _dtypes.qint32, _dtypes.bfloat16, _dtypes.qint16, _dtypes.quint16, _dtypes.uint16, _dtypes.complex128, _dtypes.half, _dtypes.uint32, _dtypes.uint64, ])
  _inputs_flat = [out_backprop]
  _attrs = ("T", _attr_T, "data_format", data_format)
  _result = _execute.execute(b"BiasAddGrad", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "BiasAddGrad", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_BiasAddV1_T = TypeVar("TV_BiasAddV1_T", _atypes.BFloat16, _atypes.Complex128, _atypes.Complex64, _atypes.Float32, _atypes.Float64, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.UInt16, _atypes.UInt32, _atypes.UInt64, _atypes.UInt8)

def bias_add_v1(value: Annotated[Any, TV_BiasAddV1_T], bias: Annotated[Any, TV_BiasAddV1_T], name=None) -> Annotated[Any, TV_BiasAddV1_T]:
  r"""Adds `bias` to `value`.

  This is a deprecated version of BiasAdd and will be soon removed.

  This is a special case of `tf.add` where `bias` is restricted to be 1-D.
  Broadcasting is supported, so `value` may have any number of dimensions.

  Args:
    value: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`, `complex64`, `int64`, `qint8`, `quint8`, `qint32`, `bfloat16`, `qint16`, `quint16`, `uint16`, `complex128`, `half`, `uint32`, `uint64`.
      Any number of dimensions.
    bias: A `Tensor`. Must have the same type as `value`.
      1-D with size the last dimension of `value`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `value`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "BiasAddV1", name, value, bias)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return bias_add_v1_eager_fallback(
          value, bias, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "BiasAddV1", value=value, bias=bias, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "BiasAddV1", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

BiasAddV1 = tf_export("raw_ops.BiasAddV1")(_ops.to_raw_op(bias_add_v1))


def bias_add_v1_eager_fallback(value: Annotated[Any, TV_BiasAddV1_T], bias: Annotated[Any, TV_BiasAddV1_T], name, ctx) -> Annotated[Any, TV_BiasAddV1_T]:
  _attr_T, _inputs_T = _execute.args_to_matching_eager([value, bias], ctx, [_dtypes.float32, _dtypes.float64, _dtypes.int32, _dtypes.uint8, _dtypes.int16, _dtypes.int8, _dtypes.complex64, _dtypes.int64, _dtypes.qint8, _dtypes.quint8, _dtypes.qint32, _dtypes.bfloat16, _dtypes.qint16, _dtypes.quint16, _dtypes.uint16, _dtypes.complex128, _dtypes.half, _dtypes.uint32, _dtypes.uint64, ])
  (value, bias) = _inputs_T
  _inputs_flat = [value, bias]
  _attrs = ("T", _attr_T)
  _result = _execute.execute(b"BiasAddV1", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "BiasAddV1", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_Conv_T = TypeVar("TV_Conv_T", _atypes.BFloat16, _atypes.Float32, _atypes.Float64, _atypes.Half, _atypes.Int32)

@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('conv')
def conv(input: Annotated[Any, TV_Conv_T], filter: Annotated[Any, TV_Conv_T], strides, padding: str, explicit_paddings=[], data_format:str="CHANNELS_LAST", dilations=[], batch_dims:int=1, groups:int=1, name=None) -> Annotated[Any, TV_Conv_T]:
  r"""Computes a N-D convolution given (N+1+batch_dims)-D `input` and (N+2)-D `filter` tensors.

  General function for computing a N-D convolution. It is required that
  `1 <= N <= 3`.

  Args:
    input: A `Tensor`. Must be one of the following types: `half`, `bfloat16`, `float32`, `float64`, `int32`.
      Tensor of type T and shape `batch_shape + spatial_shape + [in_channels]` in the
      case that `channels_last_format = true` or shape
      `batch_shape + [in_channels] + spatial_shape` if `channels_last_format = false`.
      spatial_shape is N-dimensional with `N=2` or `N=3`.
      Also note that `batch_shape` is dictated by the parameter `batch_dims`
      and defaults to 1.
    filter: A `Tensor`. Must have the same type as `input`.
      An `(N+2)-D` Tensor with the same type as `input` and shape
      `spatial_filter_shape + [in_channels, out_channels]`, where spatial_filter_shape
      is N-dimensional with `N=2` or `N=3`.
    strides: A list of `ints`.
      1-D tensor of length `N+2`. The stride of the sliding window for each
      dimension of `input`. Must have `strides[0] = strides[N+1] = 1`.
    padding: A `string` from: `"SAME", "VALID", "EXPLICIT"`.
      The type of padding algorithm to use.
    explicit_paddings: An optional list of `ints`. Defaults to `[]`.
      If `padding` is `"EXPLICIT"`, the list of explicit padding amounts. For the ith
      dimension, the amount of padding inserted before and after the dimension is
      `explicit_paddings[2 * i]` and `explicit_paddings[2 * i + 1]`, respectively. If
      `padding` is not `"EXPLICIT"`, `explicit_paddings` must be empty.
    data_format: An optional `string` from: `"CHANNELS_FIRST", "CHANNELS_LAST"`. Defaults to `"CHANNELS_LAST"`.
      Used to set the data format. By default `CHANNELS_FIRST`, uses 
      `NHWC (2D) / NDHWC (3D)` or if `CHANNELS_LAST`, uses `NCHW (2D) / NCDHW (3D)`.
    dilations: An optional list of `ints`. Defaults to `[]`.
      1-D tensor of length `N+2`. The dilation factor for each dimension of
      `input`. If set to `k > 1`, there will be `k-1` skipped cells between each
      filter element on that dimension. The dimension order is determined by the
      value of `channels_last_format`, see above for details. Dilations in the batch
      and depth dimensions must be 1.
    batch_dims: An optional `int`. Defaults to `1`.
      A positive integer specifying the number of batch dimensions for the input
      tensor. Should be less than the rank of the input tensor.
    groups: An optional `int`. Defaults to `1`.
      A positive integer specifying the number of groups in which the input is split
      along the channel axis. Each group is convolved separately with
      `filters / groups` filters. The output is the concatenation of all the groups
      results along the channel axis. Input channels and filters must both be
      divisible by groups.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "Conv", name, input, filter, "strides", strides, "padding",
        padding, "explicit_paddings", explicit_paddings, "data_format",
        data_format, "dilations", dilations, "batch_dims", batch_dims,
        "groups", groups)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_conv(
          (input, filter, strides, padding, explicit_paddings, data_format,
          dilations, batch_dims, groups, name,), None)
      if _result is not NotImplemented:
        return _result
      return conv_eager_fallback(
          input, filter, strides=strides, padding=padding,
          explicit_paddings=explicit_paddings, data_format=data_format,
          dilations=dilations, batch_dims=batch_dims, groups=groups,
          name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            conv, (), dict(input=input, filter=filter, strides=strides,
                           padding=padding,
                           explicit_paddings=explicit_paddings,
                           data_format=data_format, dilations=dilations,
                           batch_dims=batch_dims, groups=groups, name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_conv(
        (input, filter, strides, padding, explicit_paddings, data_format,
        dilations, batch_dims, groups, name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  if not isinstance(strides, (list, tuple)):
    raise TypeError(
        "Expected list for 'strides' argument to "
        "'conv' Op, not %r." % strides)
  strides = [_execute.make_int(_i, "strides") for _i in strides]
  padding = _execute.make_str(padding, "padding")
  if explicit_paddings is None:
    explicit_paddings = []
  if not isinstance(explicit_paddings, (list, tuple)):
    raise TypeError(
        "Expected list for 'explicit_paddings' argument to "
        "'conv' Op, not %r." % explicit_paddings)
  explicit_paddings = [_execute.make_int(_i, "explicit_paddings") for _i in explicit_paddings]
  if data_format is None:
    data_format = "CHANNELS_LAST"
  data_format = _execute.make_str(data_format, "data_format")
  if dilations is None:
    dilations = []
  if not isinstance(dilations, (list, tuple)):
    raise TypeError(
        "Expected list for 'dilations' argument to "
        "'conv' Op, not %r." % dilations)
  dilations = [_execute.make_int(_i, "dilations") for _i in dilations]
  if batch_dims is None:
    batch_dims = 1
  batch_dims = _execute.make_int(batch_dims, "batch_dims")
  if groups is None:
    groups = 1
  groups = _execute.make_int(groups, "groups")
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "Conv", input=input, filter=filter, strides=strides, padding=padding,
                explicit_paddings=explicit_paddings, data_format=data_format,
                dilations=dilations, batch_dims=batch_dims, groups=groups,
                name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          conv, (), dict(input=input, filter=filter, strides=strides,
                         padding=padding, explicit_paddings=explicit_paddings,
                         data_format=data_format, dilations=dilations,
                         batch_dims=batch_dims, groups=groups, name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"), "strides",
              _op.get_attr("strides"), "padding", _op.get_attr("padding"),
              "explicit_paddings", _op.get_attr("explicit_paddings"),
              "data_format", _op.get_attr("data_format"), "dilations",
              _op.get_attr("dilations"), "batch_dims",
              _op._get_attr_int("batch_dims"), "groups",
              _op._get_attr_int("groups"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "Conv", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

Conv = tf_export("raw_ops.Conv")(_ops.to_raw_op(conv))
_dispatcher_for_conv = conv._tf_type_based_dispatcher.Dispatch


def conv_eager_fallback(input: Annotated[Any, TV_Conv_T], filter: Annotated[Any, TV_Conv_T], strides, padding: str, explicit_paddings, data_format: str, dilations, batch_dims: int, groups: int, name, ctx) -> Annotated[Any, TV_Conv_T]:
  if not isinstance(strides, (list, tuple)):
    raise TypeError(
        "Expected list for 'strides' argument to "
        "'conv' Op, not %r." % strides)
  strides = [_execute.make_int(_i, "strides") for _i in strides]
  padding = _execute.make_str(padding, "padding")
  if explicit_paddings is None:
    explicit_paddings = []
  if not isinstance(explicit_paddings, (list, tuple)):
    raise TypeError(
        "Expected list for 'explicit_paddings' argument to "
        "'conv' Op, not %r." % explicit_paddings)
  explicit_paddings = [_execute.make_int(_i, "explicit_paddings") for _i in explicit_paddings]
  if data_format is None:
    data_format = "CHANNELS_LAST"
  data_format = _execute.make_str(data_format, "data_format")
  if dilations is None:
    dilations = []
  if not isinstance(dilations, (list, tuple)):
    raise TypeError(
        "Expected list for 'dilations' argument to "
        "'conv' Op, not %r." % dilations)
  dilations = [_execute.make_int(_i, "dilations") for _i in dilations]
  if batch_dims is None:
    batch_dims = 1
  batch_dims = _execute.make_int(batch_dims, "batch_dims")
  if groups is None:
    groups = 1
  groups = _execute.make_int(groups, "groups")
  _attr_T, _inputs_T = _execute.args_to_matching_eager([input, filter], ctx, [_dtypes.half, _dtypes.bfloat16, _dtypes.float32, _dtypes.float64, _dtypes.int32, ])
  (input, filter) = _inputs_T
  _inputs_flat = [input, filter]
  _attrs = ("T", _attr_T, "strides", strides, "padding", padding,
  "explicit_paddings", explicit_paddings, "data_format", data_format,
  "dilations", dilations, "batch_dims", batch_dims, "groups", groups)
  _result = _execute.execute(b"Conv", 1, inputs=_inputs_flat, attrs=_attrs,
                             ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "Conv", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_Conv2D_T = TypeVar("TV_Conv2D_T", _atypes.BFloat16, _atypes.Float32, _atypes.Float64, _atypes.Half, _atypes.Int32)

def conv2d(input: Annotated[Any, TV_Conv2D_T], filter: Annotated[Any, TV_Conv2D_T], strides, padding: str, use_cudnn_on_gpu:bool=True, explicit_paddings=[], data_format:str="NHWC", dilations=[1, 1, 1, 1], name=None) -> Annotated[Any, TV_Conv2D_T]:
  r"""Computes a 2-D convolution given 4-D `input` and `filter` tensors.

  Given an input tensor of shape `[batch, in_height, in_width, in_channels]`
  and a filter / kernel tensor of shape
  `[filter_height, filter_width, in_channels, out_channels]`, this op
  performs the following:

  1. Flattens the filter to a 2-D matrix with shape
     `[filter_height * filter_width * in_channels, output_channels]`.
  2. Extracts image patches from the input tensor to form a *virtual*
     tensor of shape `[batch, out_height, out_width,
     filter_height * filter_width * in_channels]`.
  3. For each patch, right-multiplies the filter matrix and the image patch
     vector.

  In detail, with the default NHWC format,

      output[b, i, j, k] =
          sum_{di, dj, q} input[b, strides[1] * i + di, strides[2] * j + dj, q] *
                          filter[di, dj, q, k]

  Must have `strides[0] = strides[3] = 1`.  For the most common case of the same
  horizontal and vertices strides, `strides = [1, stride, stride, 1]`.

  Args:
    input: A `Tensor`. Must be one of the following types: `half`, `bfloat16`, `float32`, `float64`, `int32`.
      A 4-D tensor. The dimension order is interpreted according to the value
      of `data_format`, see below for details.
    filter: A `Tensor`. Must have the same type as `input`.
      A 4-D tensor of shape
      `[filter_height, filter_width, in_channels, out_channels]`
    strides: A list of `ints`.
      1-D tensor of length 4.  The stride of the sliding window for each
      dimension of `input`. The dimension order is determined by the value of
      `data_format`, see below for details.
    padding: A `string` from: `"SAME", "VALID", "EXPLICIT"`.
      The type of padding algorithm to use.
    use_cudnn_on_gpu: An optional `bool`. Defaults to `True`.
    explicit_paddings: An optional list of `ints`. Defaults to `[]`.
      If `padding` is `"EXPLICIT"`, the list of explicit padding amounts. For the ith
      dimension, the amount of padding inserted before and after the dimension is
      `explicit_paddings[2 * i]` and `explicit_paddings[2 * i + 1]`, respectively. If
      `padding` is not `"EXPLICIT"`, `explicit_paddings` must be empty.
    data_format: An optional `string` from: `"NHWC", "NCHW"`. Defaults to `"NHWC"`.
      Specify the data format of the input and output data. With the
      default format "NHWC", the data is stored in the order of:
          [batch, height, width, channels].
      Alternatively, the format could be "NCHW", the data storage order of:
          [batch, channels, height, width].
    dilations: An optional list of `ints`. Defaults to `[1, 1, 1, 1]`.
      1-D tensor of length 4.  The dilation factor for each dimension of
      `input`. If set to k > 1, there will be k-1 skipped cells between each
      filter element on that dimension. The dimension order is determined by the
      value of `data_format`, see above for details. Dilations in the batch and
      depth dimensions must be 1.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "Conv2D", name, input, filter, "strides", strides,
        "use_cudnn_on_gpu", use_cudnn_on_gpu, "padding", padding,
        "explicit_paddings", explicit_paddings, "data_format", data_format,
        "dilations", dilations)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return conv2d_eager_fallback(
          input, filter, strides=strides, use_cudnn_on_gpu=use_cudnn_on_gpu,
          padding=padding, explicit_paddings=explicit_paddings,
          data_format=data_format, dilations=dilations, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if not isinstance(strides, (list, tuple)):
    raise TypeError(
        "Expected list for 'strides' argument to "
        "'conv2d' Op, not %r." % strides)
  strides = [_execute.make_int(_i, "strides") for _i in strides]
  padding = _execute.make_str(padding, "padding")
  if use_cudnn_on_gpu is None:
    use_cudnn_on_gpu = True
  use_cudnn_on_gpu = _execute.make_bool(use_cudnn_on_gpu, "use_cudnn_on_gpu")
  if explicit_paddings is None:
    explicit_paddings = []
  if not isinstance(explicit_paddings, (list, tuple)):
    raise TypeError(
        "Expected list for 'explicit_paddings' argument to "
        "'conv2d' Op, not %r." % explicit_paddings)
  explicit_paddings = [_execute.make_int(_i, "explicit_paddings") for _i in explicit_paddings]
  if data_format is None:
    data_format = "NHWC"
  data_format = _execute.make_str(data_format, "data_format")
  if dilations is None:
    dilations = [1, 1, 1, 1]
  if not isinstance(dilations, (list, tuple)):
    raise TypeError(
        "Expected list for 'dilations' argument to "
        "'conv2d' Op, not %r." % dilations)
  dilations = [_execute.make_int(_i, "dilations") for _i in dilations]
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "Conv2D", input=input, filter=filter, strides=strides,
                  padding=padding, use_cudnn_on_gpu=use_cudnn_on_gpu,
                  explicit_paddings=explicit_paddings,
                  data_format=data_format, dilations=dilations, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"), "strides",
              _op.get_attr("strides"), "use_cudnn_on_gpu",
              _op._get_attr_bool("use_cudnn_on_gpu"), "padding",
              _op.get_attr("padding"), "explicit_paddings",
              _op.get_attr("explicit_paddings"), "data_format",
              _op.get_attr("data_format"), "dilations",
              _op.get_attr("dilations"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "Conv2D", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

Conv2D = tf_export("raw_ops.Conv2D")(_ops.to_raw_op(conv2d))


def conv2d_eager_fallback(input: Annotated[Any, TV_Conv2D_T], filter: Annotated[Any, TV_Conv2D_T], strides, padding: str, use_cudnn_on_gpu: bool, explicit_paddings, data_format: str, dilations, name, ctx) -> Annotated[Any, TV_Conv2D_T]:
  if not isinstance(strides, (list, tuple)):
    raise TypeError(
        "Expected list for 'strides' argument to "
        "'conv2d' Op, not %r." % strides)
  strides = [_execute.make_int(_i, "strides") for _i in strides]
  padding = _execute.make_str(padding, "padding")
  if use_cudnn_on_gpu is None:
    use_cudnn_on_gpu = True
  use_cudnn_on_gpu = _execute.make_bool(use_cudnn_on_gpu, "use_cudnn_on_gpu")
  if explicit_paddings is None:
    explicit_paddings = []
  if not isinstance(explicit_paddings, (list, tuple)):
    raise TypeError(
        "Expected list for 'explicit_paddings' argument to "
        "'conv2d' Op, not %r." % explicit_paddings)
  explicit_paddings = [_execute.make_int(_i, "explicit_paddings") for _i in explicit_paddings]
  if data_format is None:
    data_format = "NHWC"
  data_format = _execute.make_str(data_format, "data_format")
  if dilations is None:
    dilations = [1, 1, 1, 1]
  if not isinstance(dilations, (list, tuple)):
    raise TypeError(
        "Expected list for 'dilations' argument to "
        "'conv2d' Op, not %r." % dilations)
  dilations = [_execute.make_int(_i, "dilations") for _i in dilations]
  _attr_T, _inputs_T = _execute.args_to_matching_eager([input, filter], ctx, [_dtypes.half, _dtypes.bfloat16, _dtypes.float32, _dtypes.float64, _dtypes.int32, ])
  (input, filter) = _inputs_T
  _inputs_flat = [input, filter]
  _attrs = ("T", _attr_T, "strides", strides, "use_cudnn_on_gpu",
  use_cudnn_on_gpu, "padding", padding, "explicit_paddings",
  explicit_paddings, "data_format", data_format, "dilations", dilations)
  _result = _execute.execute(b"Conv2D", 1, inputs=_inputs_flat, attrs=_attrs,
                             ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "Conv2D", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_Conv2DBackpropFilter_T = TypeVar("TV_Conv2DBackpropFilter_T", _atypes.BFloat16, _atypes.Float32, _atypes.Float64, _atypes.Half)

def conv2d_backprop_filter(input: Annotated[Any, TV_Conv2DBackpropFilter_T], filter_sizes: Annotated[Any, _atypes.Int32], out_backprop: Annotated[Any, TV_Conv2DBackpropFilter_T], strides, padding: str, use_cudnn_on_gpu:bool=True, explicit_paddings=[], data_format:str="NHWC", dilations=[1, 1, 1, 1], name=None) -> Annotated[Any, TV_Conv2DBackpropFilter_T]:
  r"""Computes the gradients of convolution with respect to the filter.

  Args:
    input: A `Tensor`. Must be one of the following types: `half`, `bfloat16`, `float32`, `float64`.
      4-D with shape `[batch, in_height, in_width, in_channels]`.
    filter_sizes: A `Tensor` of type `int32`.
      An integer vector representing the tensor shape of `filter`,
      where `filter` is a 4-D
      `[filter_height, filter_width, in_channels, out_channels]` tensor.
    out_backprop: A `Tensor`. Must have the same type as `input`.
      4-D with shape `[batch, out_height, out_width, out_channels]`.
      Gradients w.r.t. the output of the convolution.
    strides: A list of `ints`.
      The stride of the sliding window for each dimension of the input
      of the convolution. Must be in the same order as the dimension specified with
      format.
    padding: A `string` from: `"SAME", "VALID", "EXPLICIT"`.
      The type of padding algorithm to use.
    use_cudnn_on_gpu: An optional `bool`. Defaults to `True`.
    explicit_paddings: An optional list of `ints`. Defaults to `[]`.
      If `padding` is `"EXPLICIT"`, the list of explicit padding amounts. For the ith
      dimension, the amount of padding inserted before and after the dimension is
      `explicit_paddings[2 * i]` and `explicit_paddings[2 * i + 1]`, respectively. If
      `padding` is not `"EXPLICIT"`, `explicit_paddings` must be empty.
    data_format: An optional `string` from: `"NHWC", "NCHW"`. Defaults to `"NHWC"`.
      Specify the data format of the input and output data. With the
      default format "NHWC", the data is stored in the order of:
          [batch, in_height, in_width, in_channels].
      Alternatively, the format could be "NCHW", the data storage order of:
          [batch, in_channels, in_height, in_width].
    dilations: An optional list of `ints`. Defaults to `[1, 1, 1, 1]`.
      1-D tensor of length 4.  The dilation factor for each dimension of
      `input`. If set to k > 1, there will be k-1 skipped cells between each filter
      element on that dimension. The dimension order is determined by the value of
      `data_format`, see above for details. Dilations in the batch and depth
      dimensions must be 1.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "Conv2DBackpropFilter", name, input, filter_sizes, out_backprop,
        "strides", strides, "use_cudnn_on_gpu", use_cudnn_on_gpu, "padding",
        padding, "explicit_paddings", explicit_paddings, "data_format",
        data_format, "dilations", dilations)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return conv2d_backprop_filter_eager_fallback(
          input, filter_sizes, out_backprop, strides=strides,
          use_cudnn_on_gpu=use_cudnn_on_gpu, padding=padding,
          explicit_paddings=explicit_paddings, data_format=data_format,
          dilations=dilations, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if not isinstance(strides, (list, tuple)):
    raise TypeError(
        "Expected list for 'strides' argument to "
        "'conv2d_backprop_filter' Op, not %r." % strides)
  strides = [_execute.make_int(_i, "strides") for _i in strides]
  padding = _execute.make_str(padding, "padding")
  if use_cudnn_on_gpu is None:
    use_cudnn_on_gpu = True
  use_cudnn_on_gpu = _execute.make_bool(use_cudnn_on_gpu, "use_cudnn_on_gpu")
  if explicit_paddings is None:
    explicit_paddings = []
  if not isinstance(explicit_paddings, (list, tuple)):
    raise TypeError(
        "Expected list for 'explicit_paddings' argument to "
        "'conv2d_backprop_filter' Op, not %r." % explicit_paddings)
  explicit_paddings = [_execute.make_int(_i, "explicit_paddings") for _i in explicit_paddings]
  if data_format is None:
    data_format = "NHWC"
  data_format = _execute.make_str(data_format, "data_format")
  if dilations is None:
    dilations = [1, 1, 1, 1]
  if not isinstance(dilations, (list, tuple)):
    raise TypeError(
        "Expected list for 'dilations' argument to "
        "'conv2d_backprop_filter' Op, not %r." % dilations)
  dilations = [_execute.make_int(_i, "dilations") for _i in dilations]
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "Conv2DBackpropFilter", input=input, filter_sizes=filter_sizes,
                                out_backprop=out_backprop, strides=strides,
                                padding=padding,
                                use_cudnn_on_gpu=use_cudnn_on_gpu,
                                explicit_paddings=explicit_paddings,
                                data_format=data_format, dilations=dilations,
                                name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"), "strides",
              _op.get_attr("strides"), "use_cudnn_on_gpu",
              _op._get_attr_bool("use_cudnn_on_gpu"), "padding",
              _op.get_attr("padding"), "explicit_paddings",
              _op.get_attr("explicit_paddings"), "data_format",
              _op.get_attr("data_format"), "dilations",
              _op.get_attr("dilations"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "Conv2DBackpropFilter", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

Conv2DBackpropFilter = tf_export("raw_ops.Conv2DBackpropFilter")(_ops.to_raw_op(conv2d_backprop_filter))


def conv2d_backprop_filter_eager_fallback(input: Annotated[Any, TV_Conv2DBackpropFilter_T], filter_sizes: Annotated[Any, _atypes.Int32], out_backprop: Annotated[Any, TV_Conv2DBackpropFilter_T], strides, padding: str, use_cudnn_on_gpu: bool, explicit_paddings, data_format: str, dilations, name, ctx) -> Annotated[Any, TV_Conv2DBackpropFilter_T]:
  if not isinstance(strides, (list, tuple)):
    raise TypeError(
        "Expected list for 'strides' argument to "
        "'conv2d_backprop_filter' Op, not %r." % strides)
  strides = [_execute.make_int(_i, "strides") for _i in strides]
  padding = _execute.make_str(padding, "padding")
  if use_cudnn_on_gpu is None:
    use_cudnn_on_gpu = True
  use_cudnn_on_gpu = _execute.make_bool(use_cudnn_on_gpu, "use_cudnn_on_gpu")
  if explicit_paddings is None:
    explicit_paddings = []
  if not isinstance(explicit_paddings, (list, tuple)):
    raise TypeError(
        "Expected list for 'explicit_paddings' argument to "
        "'conv2d_backprop_filter' Op, not %r." % explicit_paddings)
  explicit_paddings = [_execute.make_int(_i, "explicit_paddings") for _i in explicit_paddings]
  if data_format is None:
    data_format = "NHWC"
  data_format = _execute.make_str(data_format, "data_format")
  if dilations is None:
    dilations = [1, 1, 1, 1]
  if not isinstance(dilations, (list, tuple)):
    raise TypeError(
        "Expected list for 'dilations' argument to "
        "'conv2d_backprop_filter' Op, not %r." % dilations)
  dilations = [_execute.make_int(_i, "dilations") for _i in dilations]
  _attr_T, _inputs_T = _execute.args_to_matching_eager([input, out_backprop], ctx, [_dtypes.half, _dtypes.bfloat16, _dtypes.float32, _dtypes.float64, ])
  (input, out_backprop) = _inputs_T
  filter_sizes = _ops.convert_to_tensor(filter_sizes, _dtypes.int32)
  _inputs_flat = [input, filter_sizes, out_backprop]
  _attrs = ("T", _attr_T, "strides", strides, "use_cudnn_on_gpu",
  use_cudnn_on_gpu, "padding", padding, "explicit_paddings",
  explicit_paddings, "data_format", data_format, "dilations", dilations)
  _result = _execute.execute(b"Conv2DBackpropFilter", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "Conv2DBackpropFilter", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_Conv2DBackpropFilterV2_T = TypeVar("TV_Conv2DBackpropFilterV2_T", _atypes.BFloat16, _atypes.Float32, _atypes.Float64, _atypes.Half)

@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('conv2d_backprop_filter_v2')
def conv2d_backprop_filter_v2(input: Annotated[Any, TV_Conv2DBackpropFilterV2_T], filter: Annotated[Any, TV_Conv2DBackpropFilterV2_T], out_backprop: Annotated[Any, TV_Conv2DBackpropFilterV2_T], strides, padding: str, use_cudnn_on_gpu:bool=True, explicit_paddings=[], data_format:str="NHWC", dilations=[1, 1, 1, 1], name=None) -> Annotated[Any, TV_Conv2DBackpropFilterV2_T]:
  r"""Computes the gradients of convolution with respect to the filter.

  Args:
    input: A `Tensor`. Must be one of the following types: `half`, `bfloat16`, `float32`, `float64`.
      4-D with shape `[batch, in_height, in_width, in_channels]`.
    filter: A `Tensor`. Must have the same type as `input`.
      4-D with shape `[filter_height, filter_width, in_channels, out_channels]`.
      Only shape of tensor is used.
    out_backprop: A `Tensor`. Must have the same type as `input`.
      4-D with shape `[batch, out_height, out_width, out_channels]`.
      Gradients w.r.t. the output of the convolution.
    strides: A list of `ints`.
      The stride of the sliding window for each dimension of the input
      of the convolution. Must be in the same order as the dimension specified with
      format.
    padding: A `string` from: `"SAME", "VALID", "EXPLICIT"`.
      The type of padding algorithm to use.
    use_cudnn_on_gpu: An optional `bool`. Defaults to `True`.
    explicit_paddings: An optional list of `ints`. Defaults to `[]`.
      If `padding` is `"EXPLICIT"`, the list of explicit padding amounts. For the ith
      dimension, the amount of padding inserted before and after the dimension is
      `explicit_paddings[2 * i]` and `explicit_paddings[2 * i + 1]`, respectively. If
      `padding` is not `"EXPLICIT"`, `explicit_paddings` must be empty.
    data_format: An optional `string` from: `"NHWC", "NCHW"`. Defaults to `"NHWC"`.
      Specify the data format of the input and output data. With the
      default format "NHWC", the data is stored in the order of:
          [batch, in_height, in_width, in_channels].
      Alternatively, the format could be "NCHW", the data storage order of:
          [batch, in_channels, in_height, in_width].
    dilations: An optional list of `ints`. Defaults to `[1, 1, 1, 1]`.
      1-D tensor of length 4.  The dilation factor for each dimension of
      `input`. If set to k > 1, there will be k-1 skipped cells between each filter
      element on that dimension. The dimension order is determined by the value of
      `data_format`, see above for details. Dilations in the batch and depth
      dimensions must be 1.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "Conv2DBackpropFilterV2", name, input, filter, out_backprop,
        "strides", strides, "use_cudnn_on_gpu", use_cudnn_on_gpu, "padding",
        padding, "explicit_paddings", explicit_paddings, "data_format",
        data_format, "dilations", dilations)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_conv2d_backprop_filter_v2(
          (input, filter, out_backprop, strides, padding, use_cudnn_on_gpu,
          explicit_paddings, data_format, dilations, name,), None)
      if _result is not NotImplemented:
        return _result
      return conv2d_backprop_filter_v2_eager_fallback(
          input, filter, out_backprop, strides=strides,
          use_cudnn_on_gpu=use_cudnn_on_gpu, padding=padding,
          explicit_paddings=explicit_paddings, data_format=data_format,
          dilations=dilations, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            conv2d_backprop_filter_v2, (), dict(input=input, filter=filter,
                                                out_backprop=out_backprop,
                                                strides=strides,
                                                padding=padding,
                                                use_cudnn_on_gpu=use_cudnn_on_gpu,
                                                explicit_paddings=explicit_paddings,
                                                data_format=data_format,
                                                dilations=dilations,
                                                name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_conv2d_backprop_filter_v2(
        (input, filter, out_backprop, strides, padding, use_cudnn_on_gpu,
        explicit_paddings, data_format, dilations, name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  if not isinstance(strides, (list, tuple)):
    raise TypeError(
        "Expected list for 'strides' argument to "
        "'conv2d_backprop_filter_v2' Op, not %r." % strides)
  strides = [_execute.make_int(_i, "strides") for _i in strides]
  padding = _execute.make_str(padding, "padding")
  if use_cudnn_on_gpu is None:
    use_cudnn_on_gpu = True
  use_cudnn_on_gpu = _execute.make_bool(use_cudnn_on_gpu, "use_cudnn_on_gpu")
  if explicit_paddings is None:
    explicit_paddings = []
  if not isinstance(explicit_paddings, (list, tuple)):
    raise TypeError(
        "Expected list for 'explicit_paddings' argument to "
        "'conv2d_backprop_filter_v2' Op, not %r." % explicit_paddings)
  explicit_paddings = [_execute.make_int(_i, "explicit_paddings") for _i in explicit_paddings]
  if data_format is None:
    data_format = "NHWC"
  data_format = _execute.make_str(data_format, "data_format")
  if dilations is None:
    dilations = [1, 1, 1, 1]
  if not isinstance(dilations, (list, tuple)):
    raise TypeError(
        "Expected list for 'dilations' argument to "
        "'conv2d_backprop_filter_v2' Op, not %r." % dilations)
  dilations = [_execute.make_int(_i, "dilations") for _i in dilations]
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "Conv2DBackpropFilterV2", input=input, filter=filter,
                                  out_backprop=out_backprop, strides=strides,
                                  padding=padding,
                                  use_cudnn_on_gpu=use_cudnn_on_gpu,
                                  explicit_paddings=explicit_paddings,
                                  data_format=data_format,
                                  dilations=dilations, name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          conv2d_backprop_filter_v2, (), dict(input=input, filter=filter,
                                              out_backprop=out_backprop,
                                              strides=strides,
                                              padding=padding,
                                              use_cudnn_on_gpu=use_cudnn_on_gpu,
                                              explicit_paddings=explicit_paddings,
                                              data_format=data_format,
                                              dilations=dilations, name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"), "strides",
              _op.get_attr("strides"), "use_cudnn_on_gpu",
              _op._get_attr_bool("use_cudnn_on_gpu"), "padding",
              _op.get_attr("padding"), "explicit_paddings",
              _op.get_attr("explicit_paddings"), "data_format",
              _op.get_attr("data_format"), "dilations",
              _op.get_attr("dilations"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "Conv2DBackpropFilterV2", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

Conv2DBackpropFilterV2 = tf_export("raw_ops.Conv2DBackpropFilterV2")(_ops.to_raw_op(conv2d_backprop_filter_v2))
_dispatcher_for_conv2d_backprop_filter_v2 = conv2d_backprop_filter_v2._tf_type_based_dispatcher.Dispatch


def conv2d_backprop_filter_v2_eager_fallback(input: Annotated[Any, TV_Conv2DBackpropFilterV2_T], filter: Annotated[Any, TV_Conv2DBackpropFilterV2_T], out_backprop: Annotated[Any, TV_Conv2DBackpropFilterV2_T], strides, padding: str, use_cudnn_on_gpu: bool, explicit_paddings, data_format: str, dilations, name, ctx) -> Annotated[Any, TV_Conv2DBackpropFilterV2_T]:
  if not isinstance(strides, (list, tuple)):
    raise TypeError(
        "Expected list for 'strides' argument to "
        "'conv2d_backprop_filter_v2' Op, not %r." % strides)
  strides = [_execute.make_int(_i, "strides") for _i in strides]
  padding = _execute.make_str(padding, "padding")
  if use_cudnn_on_gpu is None:
    use_cudnn_on_gpu = True
  use_cudnn_on_gpu = _execute.make_bool(use_cudnn_on_gpu, "use_cudnn_on_gpu")
  if explicit_paddings is None:
    explicit_paddings = []
  if not isinstance(explicit_paddings, (list, tuple)):
    raise TypeError(
        "Expected list for 'explicit_paddings' argument to "
        "'conv2d_backprop_filter_v2' Op, not %r." % explicit_paddings)
  explicit_paddings = [_execute.make_int(_i, "explicit_paddings") for _i in explicit_paddings]
  if data_format is None:
    data_format = "NHWC"
  data_format = _execute.make_str(data_format, "data_format")
  if dilations is None:
    dilations = [1, 1, 1, 1]
  if not isinstance(dilations, (list, tuple)):
    raise TypeError(
        "Expected list for 'dilations' argument to "
        "'conv2d_backprop_filter_v2' Op, not %r." % dilations)
  dilations = [_execute.make_int(_i, "dilations") for _i in dilations]
  _attr_T, _inputs_T = _execute.args_to_matching_eager([input, filter, out_backprop], ctx, [_dtypes.half, _dtypes.bfloat16, _dtypes.float32, _dtypes.float64, ])
  (input, filter, out_backprop) = _inputs_T
  _inputs_flat = [input, filter, out_backprop]
  _attrs = ("T", _attr_T, "strides", strides, "use_cudnn_on_gpu",
  use_cudnn_on_gpu, "padding", padding, "explicit_paddings",
  explicit_paddings, "data_format", data_format, "dilations", dilations)
  _result = _execute.execute(b"Conv2DBackpropFilterV2", 1,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "Conv2DBackpropFilterV2", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_Conv2DBackpropInput_T = TypeVar("TV_Conv2DBackpropInput_T", _atypes.BFloat16, _atypes.Float32, _atypes.Float64, _atypes.Half, _atypes.Int32)

def conv2d_backprop_input(input_sizes: Annotated[Any, _atypes.Int32], filter: Annotated[Any, TV_Conv2DBackpropInput_T], out_backprop: Annotated[Any, TV_Conv2DBackpropInput_T], strides, padding: str, use_cudnn_on_gpu:bool=True, explicit_paddings=[], data_format:str="NHWC", dilations=[1, 1, 1, 1], name=None) -> Annotated[Any, TV_Conv2DBackpropInput_T]:
  r"""Computes the gradients of convolution with respect to the input.

  Args:
    input_sizes: A `Tensor` of type `int32`.
      An integer vector representing the shape of `input`,
      where `input` is a 4-D `[batch, height, width, channels]` tensor.
    filter: A `Tensor`. Must be one of the following types: `half`, `bfloat16`, `float32`, `float64`, `int32`.
      4-D with shape
      `[filter_height, filter_width, in_channels, out_channels]`.
    out_backprop: A `Tensor`. Must have the same type as `filter`.
      4-D with shape `[batch, out_height, out_width, out_channels]`.
      Gradients w.r.t. the output of the convolution.
    strides: A list of `ints`.
      The stride of the sliding window for each dimension of the input
      of the convolution. Must be in the same order as the dimension specified with
      format.
    padding: A `string` from: `"SAME", "VALID", "EXPLICIT"`.
      The type of padding algorithm to use.
    use_cudnn_on_gpu: An optional `bool`. Defaults to `True`.
    explicit_paddings: An optional list of `ints`. Defaults to `[]`.
      If `padding` is `"EXPLICIT"`, the list of explicit padding amounts. For the ith
      dimension, the amount of padding inserted before and after the dimension is
      `explicit_paddings[2 * i]` and `explicit_paddings[2 * i + 1]`, respectively. If
      `padding` is not `"EXPLICIT"`, `explicit_paddings` must be empty.
    data_format: An optional `string` from: `"NHWC", "NCHW"`. Defaults to `"NHWC"`.
      Specify the data format of the input and output data. With the
      default format "NHWC", the data is stored in the order of:
          [batch, in_height, in_width, in_channels].
      Alternatively, the format could be "NCHW", the data storage order of:
          [batch, in_channels, in_height, in_width].
    dilations: An optional list of `ints`. Defaults to `[1, 1, 1, 1]`.
      1-D tensor of length 4.  The dilation factor for each dimension of
      `input`. If set to k > 1, there will be k-1 skipped cells between each filter
      element on that dimension. The dimension order is determined by the value of
      `data_format`, see above for details. Dilations in the batch and depth
      dimensions must be 1.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `filter`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "Conv2DBackpropInput", name, input_sizes, filter, out_backprop,
        "strides", strides, "use_cudnn_on_gpu", use_cudnn_on_gpu, "padding",
        padding, "explicit_paddings", explicit_paddings, "data_format",
        data_format, "dilations", dilations)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return conv2d_backprop_input_eager_fallback(
          input_sizes, filter, out_backprop, strides=strides,
          use_cudnn_on_gpu=use_cudnn_on_gpu, padding=padding,
          explicit_paddings=explicit_paddings, data_format=data_format,
          dilations=dilations, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if not isinstance(strides, (list, tuple)):
    raise TypeError(
        "Expected list for 'strides' argument to "
        "'conv2d_backprop_input' Op, not %r." % strides)
  strides = [_execute.make_int(_i, "strides") for _i in strides]
  padding = _execute.make_str(padding, "padding")
  if use_cudnn_on_gpu is None:
    use_cudnn_on_gpu = True
  use_cudnn_on_gpu = _execute.make_bool(use_cudnn_on_gpu, "use_cudnn_on_gpu")
  if explicit_paddings is None:
    explicit_paddings = []
  if not isinstance(explicit_paddings, (list, tuple)):
    raise TypeError(
        "Expected list for 'explicit_paddings' argument to "
        "'conv2d_backprop_input' Op, not %r." % explicit_paddings)
  explicit_paddings = [_execute.make_int(_i, "explicit_paddings") for _i in explicit_paddings]
  if data_format is None:
    data_format = "NHWC"
  data_format = _execute.make_str(data_format, "data_format")
  if dilations is None:
    dilations = [1, 1, 1, 1]
  if not isinstance(dilations, (list, tuple)):
    raise TypeError(
        "Expected list for 'dilations' argument to "
        "'conv2d_backprop_input' Op, not %r." % dilations)
  dilations = [_execute.make_int(_i, "dilations") for _i in dilations]
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "Conv2DBackpropInput", input_sizes=input_sizes, filter=filter,
                               out_backprop=out_backprop, strides=strides,
                               padding=padding,
                               use_cudnn_on_gpu=use_cudnn_on_gpu,
                               explicit_paddings=explicit_paddings,
                               data_format=data_format, dilations=dilations,
                               name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"), "strides",
              _op.get_attr("strides"), "use_cudnn_on_gpu",
              _op._get_attr_bool("use_cudnn_on_gpu"), "padding",
              _op.get_attr("padding"), "explicit_paddings",
              _op.get_attr("explicit_paddings"), "data_format",
              _op.get_attr("data_format"), "dilations",
              _op.get_attr("dilations"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "Conv2DBackpropInput", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

Conv2DBackpropInput = tf_export("raw_ops.Conv2DBackpropInput")(_ops.to_raw_op(conv2d_backprop_input))


def conv2d_backprop_input_eager_fallback(input_sizes: Annotated[Any, _atypes.Int32], filter: Annotated[Any, TV_Conv2DBackpropInput_T], out_backprop: Annotated[Any, TV_Conv2DBackpropInput_T], strides, padding: str, use_cudnn_on_gpu: bool, explicit_paddings, data_format: str, dilations, name, ctx) -> Annotated[Any, TV_Conv2DBackpropInput_T]:
  if not isinstance(strides, (list, tuple)):
    raise TypeError(
        "Expected list for 'strides' argument to "
        "'conv2d_backprop_input' Op, not %r." % strides)
  strides = [_execute.make_int(_i, "strides") for _i in strides]
  padding = _execute.make_str(padding, "padding")
  if use_cudnn_on_gpu is None:
    use_cudnn_on_gpu = True
  use_cudnn_on_gpu = _execute.make_bool(use_cudnn_on_gpu, "use_cudnn_on_gpu")
  if explicit_paddings is None:
    explicit_paddings = []
  if not isinstance(explicit_paddings, (list, tuple)):
    raise TypeError(
        "Expected list for 'explicit_paddings' argument to "
        "'conv2d_backprop_input' Op, not %r." % explicit_paddings)
  explicit_paddings = [_execute.make_int(_i, "explicit_paddings") for _i in explicit_paddings]
  if data_format is None:
    data_format = "NHWC"
  data_format = _execute.make_str(data_format, "data_format")
  if dilations is None:
    dilations = [1, 1, 1, 1]
  if not isinstance(dilations, (list, tuple)):
    raise TypeError(
        "Expected list for 'dilations' argument to "
        "'conv2d_backprop_input' Op, not %r." % dilations)
  dilations = [_execute.make_int(_i, "dilations") for _i in dilations]
  _attr_T, _inputs_T = _execute.args_to_matching_eager([filter, out_backprop], ctx, [_dtypes.half, _dtypes.bfloat16, _dtypes.float32, _dtypes.float64, _dtypes.int32, ])
  (filter, out_backprop) = _inputs_T
  input_sizes = _ops.convert_to_tensor(input_sizes, _dtypes.int32)
  _inputs_flat = [input_sizes, filter, out_backprop]
  _attrs = ("T", _attr_T, "strides", strides, "use_cudnn_on_gpu",
  use_cudnn_on_gpu, "padding", padding, "explicit_paddings",
  explicit_paddings, "data_format", data_format, "dilations", dilations)
  _result = _execute.execute(b"Conv2DBackpropInput", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "Conv2DBackpropInput", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_Conv2DBackpropInputV2_T = TypeVar("TV_Conv2DBackpropInputV2_T", _atypes.BFloat16, _atypes.Float32, _atypes.Float64, _atypes.Half, _atypes.Int32)

@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('conv2d_backprop_input_v2')
def conv2d_backprop_input_v2(input: Annotated[Any, TV_Conv2DBackpropInputV2_T], filter: Annotated[Any, TV_Conv2DBackpropInputV2_T], out_backprop: Annotated[Any, TV_Conv2DBackpropInputV2_T], strides, padding: str, use_cudnn_on_gpu:bool=True, explicit_paddings=[], data_format:str="NHWC", dilations=[1, 1, 1, 1], name=None) -> Annotated[Any, TV_Conv2DBackpropInputV2_T]:
  r"""Computes the gradients of convolution with respect to the input.

  Args:
    input: A `Tensor`. Must be one of the following types: `half`, `bfloat16`, `float32`, `float64`, `int32`.
      4-D with shape `[batch, in_height, in_width, in_channels]`.
      Only shape of tensor is used.
    filter: A `Tensor`. Must have the same type as `input`. 4-D with shape
      `[filter_height, filter_width, in_channels, out_channels]`.
    out_backprop: A `Tensor`. Must have the same type as `input`.
      4-D with shape `[batch, out_height, out_width, out_channels]`.
      Gradients w.r.t. the output of the convolution.
    strides: A list of `ints`.
      The stride of the sliding window for each dimension of the input
      of the convolution. Must be in the same order as the dimension specified with
      format.
    padding: A `string` from: `"SAME", "VALID", "EXPLICIT"`.
      The type of padding algorithm to use.
    use_cudnn_on_gpu: An optional `bool`. Defaults to `True`.
    explicit_paddings: An optional list of `ints`. Defaults to `[]`.
      If `padding` is `"EXPLICIT"`, the list of explicit padding amounts. For the ith
      dimension, the amount of padding inserted before and after the dimension is
      `explicit_paddings[2 * i]` and `explicit_paddings[2 * i + 1]`, respectively. If
      `padding` is not `"EXPLICIT"`, `explicit_paddings` must be empty.
    data_format: An optional `string` from: `"NHWC", "NCHW"`. Defaults to `"NHWC"`.
      Specify the data format of the input and output data. With the
      default format "NHWC", the data is stored in the order of:
          [batch, in_height, in_width, in_channels].
      Alternatively, the format could be "NCHW", the data storage order of:
          [batch, in_channels, in_height, in_width].
    dilations: An optional list of `ints`. Defaults to `[1, 1, 1, 1]`.
      1-D tensor of length 4.  The dilation factor for each dimension of
      `input`. If set to k > 1, there will be k-1 skipped cells between each filter
      element on that dimension. The dimension order is determined by the value of
      `data_format`, see above for details. Dilations in the batch and depth
      dimensions must be 1.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "Conv2DBackpropInputV2", name, input, filter, out_backprop,
        "strides", strides, "use_cudnn_on_gpu", use_cudnn_on_gpu, "padding",
        padding, "explicit_paddings", explicit_paddings, "data_format",
        data_format, "dilations", dilations)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_conv2d_backprop_input_v2(
          (input, filter, out_backprop, strides, padding, use_cudnn_on_gpu,
          explicit_paddings, data_format, dilations, name,), None)
      if _result is not NotImplemented:
        return _result
      return conv2d_backprop_input_v2_eager_fallback(
          input, filter, out_backprop, strides=strides,
          use_cudnn_on_gpu=use_cudnn_on_gpu, padding=padding,
          explicit_paddings=explicit_paddings, data_format=data_format,
          dilations=dilations, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            conv2d_backprop_input_v2, (), dict(input=input, filter=filter,
                                               out_backprop=out_backprop,
                                               strides=strides,
                                               padding=padding,
                                               use_cudnn_on_gpu=use_cudnn_on_gpu,
                                               explicit_paddings=explicit_paddings,
                                               data_format=data_format,
                                               dilations=dilations, name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_conv2d_backprop_input_v2(
        (input, filter, out_backprop, strides, padding, use_cudnn_on_gpu,
        explicit_paddings, data_format, dilations, name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  if not isinstance(strides, (list, tuple)):
    raise TypeError(
        "Expected list for 'strides' argument to "
        "'conv2d_backprop_input_v2' Op, not %r." % strides)
  strides = [_execute.make_int(_i, "strides") for _i in strides]
  padding = _execute.make_str(padding, "padding")
  if use_cudnn_on_gpu is None:
    use_cudnn_on_gpu = True
  use_cudnn_on_gpu = _execute.make_bool(use_cudnn_on_gpu, "use_cudnn_on_gpu")
  if explicit_paddings is None:
    explicit_paddings = []
  if not isinstance(explicit_paddings, (list, tuple)):
    raise TypeError(
        "Expected list for 'explicit_paddings' argument to "
        "'conv2d_backprop_input_v2' Op, not %r." % explicit_paddings)
  explicit_paddings = [_execute.make_int(_i, "explicit_paddings") for _i in explicit_paddings]
  if data_format is None:
    data_format = "NHWC"
  data_format = _execute.make_str(data_format, "data_format")
  if dilations is None:
    dilations = [1, 1, 1, 1]
  if not isinstance(dilations, (list, tuple)):
    raise TypeError(
        "Expected list for 'dilations' argument to "
        "'conv2d_backprop_input_v2' Op, not %r." % dilations)
  dilations = [_execute.make_int(_i, "dilations") for _i in dilations]
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "Conv2DBackpropInputV2", input=input, filter=filter,
                                 out_backprop=out_backprop, strides=strides,
                                 padding=padding,
                                 use_cudnn_on_gpu=use_cudnn_on_gpu,
                                 explicit_paddings=explicit_paddings,
                                 data_format=data_format, dilations=dilations,
                                 name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          conv2d_backprop_input_v2, (), dict(input=input, filter=filter,
                                             out_backprop=out_backprop,
                                             strides=strides, padding=padding,
                                             use_cudnn_on_gpu=use_cudnn_on_gpu,
                                             explicit_paddings=explicit_paddings,
                                             data_format=data_format,
                                             dilations=dilations, name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"), "strides",
              _op.get_attr("strides"), "use_cudnn_on_gpu",
              _op._get_attr_bool("use_cudnn_on_gpu"), "padding",
              _op.get_attr("padding"), "explicit_paddings",
              _op.get_attr("explicit_paddings"), "data_format",
              _op.get_attr("data_format"), "dilations",
              _op.get_attr("dilations"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "Conv2DBackpropInputV2", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

Conv2DBackpropInputV2 = tf_export("raw_ops.Conv2DBackpropInputV2")(_ops.to_raw_op(conv2d_backprop_input_v2))
_dispatcher_for_conv2d_backprop_input_v2 = conv2d_backprop_input_v2._tf_type_based_dispatcher.Dispatch


def conv2d_backprop_input_v2_eager_fallback(input: Annotated[Any, TV_Conv2DBackpropInputV2_T], filter: Annotated[Any, TV_Conv2DBackpropInputV2_T], out_backprop: Annotated[Any, TV_Conv2DBackpropInputV2_T], strides, padding: str, use_cudnn_on_gpu: bool, explicit_paddings, data_format: str, dilations, name, ctx) -> Annotated[Any, TV_Conv2DBackpropInputV2_T]:
  if not isinstance(strides, (list, tuple)):
    raise TypeError(
        "Expected list for 'strides' argument to "
        "'conv2d_backprop_input_v2' Op, not %r." % strides)
  strides = [_execute.make_int(_i, "strides") for _i in strides]
  padding = _execute.make_str(padding, "padding")
  if use_cudnn_on_gpu is None:
    use_cudnn_on_gpu = True
  use_cudnn_on_gpu = _execute.make_bool(use_cudnn_on_gpu, "use_cudnn_on_gpu")
  if explicit_paddings is None:
    explicit_paddings = []
  if not isinstance(explicit_paddings, (list, tuple)):
    raise TypeError(
        "Expected list for 'explicit_paddings' argument to "
        "'conv2d_backprop_input_v2' Op, not %r." % explicit_paddings)
  explicit_paddings = [_execute.make_int(_i, "explicit_paddings") for _i in explicit_paddings]
  if data_format is None:
    data_format = "NHWC"
  data_format = _execute.make_str(data_format, "data_format")
  if dilations is None:
    dilations = [1, 1, 1, 1]
  if not isinstance(dilations, (list, tuple)):
    raise TypeError(
        "Expected list for 'dilations' argument to "
        "'conv2d_backprop_input_v2' Op, not %r." % dilations)
  dilations = [_execute.make_int(_i, "dilations") for _i in dilations]
  _attr_T, _inputs_T = _execute.args_to_matching_eager([input, filter, out_backprop], ctx, [_dtypes.half, _dtypes.bfloat16, _dtypes.float32, _dtypes.float64, _dtypes.int32, ])
  (input, filter, out_backprop) = _inputs_T
  _inputs_flat = [input, filter, out_backprop]
  _attrs = ("T", _attr_T, "strides", strides, "use_cudnn_on_gpu",
  use_cudnn_on_gpu, "padding", padding, "explicit_paddings",
  explicit_paddings, "data_format", data_format, "dilations", dilations)
  _result = _execute.execute(b"Conv2DBackpropInputV2", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "Conv2DBackpropInputV2", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_Conv3D_T = TypeVar("TV_Conv3D_T", _atypes.BFloat16, _atypes.Float32, _atypes.Float64, _atypes.Half)

def conv3d(input: Annotated[Any, TV_Conv3D_T], filter: Annotated[Any, TV_Conv3D_T], strides, padding: str, data_format:str="NDHWC", dilations=[1, 1, 1, 1, 1], name=None) -> Annotated[Any, TV_Conv3D_T]:
  r"""Computes a 3-D convolution given 5-D `input` and `filter` tensors.

  In signal processing, cross-correlation is a measure of similarity of
  two waveforms as a function of a time-lag applied to one of them. This
  is also known as a sliding dot product or sliding inner-product.

  Our Conv3D implements a form of cross-correlation.

  Args:
    input: A `Tensor`. Must be one of the following types: `half`, `bfloat16`, `float32`, `float64`.
      Shape `[batch, in_depth, in_height, in_width, in_channels]`.
    filter: A `Tensor`. Must have the same type as `input`.
      Shape `[filter_depth, filter_height, filter_width, in_channels,
      out_channels]`. `in_channels` must match between `input` and `filter`.
    strides: A list of `ints` that has length `>= 5`.
      1-D tensor of length 5. The stride of the sliding window for each
      dimension of `input`. Must have `strides[0] = strides[4] = 1`.
    padding: A `string` from: `"SAME", "VALID"`.
      The type of padding algorithm to use.
    data_format: An optional `string` from: `"NDHWC", "NCDHW"`. Defaults to `"NDHWC"`.
      The data format of the input and output data. With the
      default format "NDHWC", the data is stored in the order of:
          [batch, in_depth, in_height, in_width, in_channels].
      Alternatively, the format could be "NCDHW", the data storage order is:
          [batch, in_channels, in_depth, in_height, in_width].
    dilations: An optional list of `ints`. Defaults to `[1, 1, 1, 1, 1]`.
      1-D tensor of length 5.  The dilation factor for each dimension of
      `input`. If set to k > 1, there will be k-1 skipped cells between each
      filter element on that dimension. The dimension order is determined by the
      value of `data_format`, see above for details. Dilations in the batch and
      depth dimensions must be 1.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "Conv3D", name, input, filter, "strides", strides, "padding",
        padding, "data_format", data_format, "dilations", dilations)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return conv3d_eager_fallback(
          input, filter, strides=strides, padding=padding,
          data_format=data_format, dilations=dilations, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if not isinstance(strides, (list, tuple)):
    raise TypeError(
        "Expected list for 'strides' argument to "
        "'conv3d' Op, not %r." % strides)
  strides = [_execute.make_int(_i, "strides") for _i in strides]
  padding = _execute.make_str(padding, "padding")
  if data_format is None:
    data_format = "NDHWC"
  data_format = _execute.make_str(data_format, "data_format")
  if dilations is None:
    dilations = [1, 1, 1, 1, 1]
  if not isinstance(dilations, (list, tuple)):
    raise TypeError(
        "Expected list for 'dilations' argument to "
        "'conv3d' Op, not %r." % dilations)
  dilations = [_execute.make_int(_i, "dilations") for _i in dilations]
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "Conv3D", input=input, filter=filter, strides=strides,
                  padding=padding, data_format=data_format,
                  dilations=dilations, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"), "strides",
              _op.get_attr("strides"), "padding", _op.get_attr("padding"),
              "data_format", _op.get_attr("data_format"), "dilations",
              _op.get_attr("dilations"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "Conv3D", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

Conv3D = tf_export("raw_ops.Conv3D")(_ops.to_raw_op(conv3d))


def conv3d_eager_fallback(input: Annotated[Any, TV_Conv3D_T], filter: Annotated[Any, TV_Conv3D_T], strides, padding: str, data_format: str, dilations, name, ctx) -> Annotated[Any, TV_Conv3D_T]:
  if not isinstance(strides, (list, tuple)):
    raise TypeError(
        "Expected list for 'strides' argument to "
        "'conv3d' Op, not %r." % strides)
  strides = [_execute.make_int(_i, "strides") for _i in strides]
  padding = _execute.make_str(padding, "padding")
  if data_format is None:
    data_format = "NDHWC"
  data_format = _execute.make_str(data_format, "data_format")
  if dilations is None:
    dilations = [1, 1, 1, 1, 1]
  if not isinstance(dilations, (list, tuple)):
    raise TypeError(
        "Expected list for 'dilations' argument to "
        "'conv3d' Op, not %r." % dilations)
  dilations = [_execute.make_int(_i, "dilations") for _i in dilations]
  _attr_T, _inputs_T = _execute.args_to_matching_eager([input, filter], ctx, [_dtypes.half, _dtypes.bfloat16, _dtypes.float32, _dtypes.float64, ])
  (input, filter) = _inputs_T
  _inputs_flat = [input, filter]
  _attrs = ("T", _attr_T, "strides", strides, "padding", padding,
  "data_format", data_format, "dilations", dilations)
  _result = _execute.execute(b"Conv3D", 1, inputs=_inputs_flat, attrs=_attrs,
                             ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "Conv3D", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_Conv3DBackpropFilter_T = TypeVar("TV_Conv3DBackpropFilter_T", _atypes.Float32, _atypes.Float64, _atypes.Half)

def conv3d_backprop_filter(input: Annotated[Any, TV_Conv3DBackpropFilter_T], filter: Annotated[Any, TV_Conv3DBackpropFilter_T], out_backprop: Annotated[Any, TV_Conv3DBackpropFilter_T], strides, padding: str, dilations=[1, 1, 1, 1, 1], name=None) -> Annotated[Any, TV_Conv3DBackpropFilter_T]:
  r"""Computes the gradients of 3-D convolution with respect to the filter.

  Args:
    input: A `Tensor`. Must be one of the following types: `half`, `float32`, `float64`.
      Shape `[batch, depth, rows, cols, in_channels]`.
    filter: A `Tensor`. Must have the same type as `input`.
      Shape `[depth, rows, cols, in_channels, out_channels]`.
      `in_channels` must match between `input` and `filter`.
    out_backprop: A `Tensor`. Must have the same type as `input`.
      Backprop signal of shape `[batch, out_depth, out_rows, out_cols,
      out_channels]`.
    strides: A list of `ints` that has length `>= 5`.
      1-D tensor of length 5. The stride of the sliding window for each
      dimension of `input`. Must have `strides[0] = strides[4] = 1`.
    padding: A `string` from: `"SAME", "VALID"`.
      The type of padding algorithm to use.
    dilations: An optional list of `ints`. Defaults to `[1, 1, 1, 1, 1]`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "Conv3DBackpropFilter", name, input, filter, out_backprop,
        "strides", strides, "padding", padding, "dilations", dilations)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return conv3d_backprop_filter_eager_fallback(
          input, filter, out_backprop, strides=strides, padding=padding,
          dilations=dilations, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if not isinstance(strides, (list, tuple)):
    raise TypeError(
        "Expected list for 'strides' argument to "
        "'conv3d_backprop_filter' Op, not %r." % strides)
  strides = [_execute.make_int(_i, "strides") for _i in strides]
  padding = _execute.make_str(padding, "padding")
  if dilations is None:
    dilations = [1, 1, 1, 1, 1]
  if not isinstance(dilations, (list, tuple)):
    raise TypeError(
        "Expected list for 'dilations' argument to "
        "'conv3d_backprop_filter' Op, not %r." % dilations)
  dilations = [_execute.make_int(_i, "dilations") for _i in dilations]
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "Conv3DBackpropFilter", input=input, filter=filter,
                                out_backprop=out_backprop, strides=strides,
                                padding=padding, dilations=dilations,
                                name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"), "strides",
              _op.get_attr("strides"), "padding", _op.get_attr("padding"),
              "dilations", _op.get_attr("dilations"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "Conv3DBackpropFilter", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

Conv3DBackpropFilter = tf_export("raw_ops.Conv3DBackpropFilter")(_ops.to_raw_op(conv3d_backprop_filter))


def conv3d_backprop_filter_eager_fallback(input: Annotated[Any, TV_Conv3DBackpropFilter_T], filter: Annotated[Any, TV_Conv3DBackpropFilter_T], out_backprop: Annotated[Any, TV_Conv3DBackpropFilter_T], strides, padding: str, dilations, name, ctx) -> Annotated[Any, TV_Conv3DBackpropFilter_T]:
  if not isinstance(strides, (list, tuple)):
    raise TypeError(
        "Expected list for 'strides' argument to "
        "'conv3d_backprop_filter' Op, not %r." % strides)
  strides = [_execute.make_int(_i, "strides") for _i in strides]
  padding = _execute.make_str(padding, "padding")
  if dilations is None:
    dilations = [1, 1, 1, 1, 1]
  if not isinstance(dilations, (list, tuple)):
    raise TypeError(
        "Expected list for 'dilations' argument to "
        "'conv3d_backprop_filter' Op, not %r." % dilations)
  dilations = [_execute.make_int(_i, "dilations") for _i in dilations]
  _attr_T, _inputs_T = _execute.args_to_matching_eager([input, filter, out_backprop], ctx, [_dtypes.half, _dtypes.float32, _dtypes.float64, ])
  (input, filter, out_backprop) = _inputs_T
  _inputs_flat = [input, filter, out_backprop]
  _attrs = ("T", _attr_T, "strides", strides, "padding", padding, "dilations",
  dilations)
  _result = _execute.execute(b"Conv3DBackpropFilter", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "Conv3DBackpropFilter", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_Conv3DBackpropFilterV2_T = TypeVar("TV_Conv3DBackpropFilterV2_T", _atypes.BFloat16, _atypes.Float32, _atypes.Float64, _atypes.Half)

@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export(v1=['nn.conv3d_backprop_filter', 'nn.conv3d_backprop_filter_v2'])
@deprecated_endpoints('nn.conv3d_backprop_filter', 'nn.conv3d_backprop_filter_v2')
def conv3d_backprop_filter_v2(input: Annotated[Any, TV_Conv3DBackpropFilterV2_T], filter_sizes: Annotated[Any, _atypes.Int32], out_backprop: Annotated[Any, TV_Conv3DBackpropFilterV2_T], strides, padding: str, data_format:str="NDHWC", dilations=[1, 1, 1, 1, 1], name=None) -> Annotated[Any, TV_Conv3DBackpropFilterV2_T]:
  r"""Computes the gradients of 3-D convolution with respect to the filter.

  Args:
    input: A `Tensor`. Must be one of the following types: `half`, `bfloat16`, `float32`, `float64`.
      Shape `[batch, depth, rows, cols, in_channels]`.
    filter_sizes: A `Tensor` of type `int32`.
      An integer vector representing the tensor shape of `filter`,
      where `filter` is a 5-D
      `[filter_depth, filter_height, filter_width, in_channels, out_channels]`
      tensor.
    out_backprop: A `Tensor`. Must have the same type as `input`.
      Backprop signal of shape `[batch, out_depth, out_rows, out_cols,
      out_channels]`.
    strides: A list of `ints` that has length `>= 5`.
      1-D tensor of length 5. The stride of the sliding window for each
      dimension of `input`. Must have `strides[0] = strides[4] = 1`.
    padding: A `string` from: `"SAME", "VALID"`.
      The type of padding algorithm to use.
    data_format: An optional `string` from: `"NDHWC", "NCDHW"`. Defaults to `"NDHWC"`.
      The data format of the input and output data. With the
      default format "NDHWC", the data is stored in the order of:
          [batch, in_depth, in_height, in_width, in_channels].
      Alternatively, the format could be "NCDHW", the data storage order is:
          [batch, in_channels, in_depth, in_height, in_width].
    dilations: An optional list of `ints`. Defaults to `[1, 1, 1, 1, 1]`.
      1-D tensor of length 5.  The dilation factor for each dimension of
      `input`. If set to k > 1, there will be k-1 skipped cells between each
      filter element on that dimension. The dimension order is determined by the
      value of `data_format`, see above for details. Dilations in the batch and
      depth dimensions must be 1.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "Conv3DBackpropFilterV2", name, input, filter_sizes,
        out_backprop, "strides", strides, "padding", padding, "data_format",
        data_format, "dilations", dilations)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_conv3d_backprop_filter_v2(
          (input, filter_sizes, out_backprop, strides, padding, data_format,
          dilations, name,), None)
      if _result is not NotImplemented:
        return _result
      return conv3d_backprop_filter_v2_eager_fallback(
          input, filter_sizes, out_backprop, strides=strides, padding=padding,
          data_format=data_format, dilations=dilations, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            conv3d_backprop_filter_v2, (), dict(input=input,
                                                filter_sizes=filter_sizes,
                                                out_backprop=out_backprop,
                                                strides=strides,
                                                padding=padding,
                                                data_format=data_format,
                                                dilations=dilations,
                                                name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_conv3d_backprop_filter_v2(
        (input, filter_sizes, out_backprop, strides, padding, data_format,
        dilations, name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  if not isinstance(strides, (list, tuple)):
    raise TypeError(
        "Expected list for 'strides' argument to "
        "'conv3d_backprop_filter_v2' Op, not %r." % strides)
  strides = [_execute.make_int(_i, "strides") for _i in strides]
  padding = _execute.make_str(padding, "padding")
  if data_format is None:
    data_format = "NDHWC"
  data_format = _execute.make_str(data_format, "data_format")
  if dilations is None:
    dilations = [1, 1, 1, 1, 1]
  if not isinstance(dilations, (list, tuple)):
    raise TypeError(
        "Expected list for 'dilations' argument to "
        "'conv3d_backprop_filter_v2' Op, not %r." % dilations)
  dilations = [_execute.make_int(_i, "dilations") for _i in dilations]
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "Conv3DBackpropFilterV2", input=input, filter_sizes=filter_sizes,
                                  out_backprop=out_backprop, strides=strides,
                                  padding=padding, data_format=data_format,
                                  dilations=dilations, name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          conv3d_backprop_filter_v2, (), dict(input=input,
                                              filter_sizes=filter_sizes,
                                              out_backprop=out_backprop,
                                              strides=strides,
                                              padding=padding,
                                              data_format=data_format,
                                              dilations=dilations, name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"), "strides",
              _op.get_attr("strides"), "padding", _op.get_attr("padding"),
              "data_format", _op.get_attr("data_format"), "dilations",
              _op.get_attr("dilations"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "Conv3DBackpropFilterV2", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

Conv3DBackpropFilterV2 = tf_export("raw_ops.Conv3DBackpropFilterV2")(_ops.to_raw_op(conv3d_backprop_filter_v2))
_dispatcher_for_conv3d_backprop_filter_v2 = conv3d_backprop_filter_v2._tf_type_based_dispatcher.Dispatch


def conv3d_backprop_filter_v2_eager_fallback(input: Annotated[Any, TV_Conv3DBackpropFilterV2_T], filter_sizes: Annotated[Any, _atypes.Int32], out_backprop: Annotated[Any, TV_Conv3DBackpropFilterV2_T], strides, padding: str, data_format: str, dilations, name, ctx) -> Annotated[Any, TV_Conv3DBackpropFilterV2_T]:
  if not isinstance(strides, (list, tuple)):
    raise TypeError(
        "Expected list for 'strides' argument to "
        "'conv3d_backprop_filter_v2' Op, not %r." % strides)
  strides = [_execute.make_int(_i, "strides") for _i in strides]
  padding = _execute.make_str(padding, "padding")
  if data_format is None:
    data_format = "NDHWC"
  data_format = _execute.make_str(data_format, "data_format")
  if dilations is None:
    dilations = [1, 1, 1, 1, 1]
  if not isinstance(dilations, (list, tuple)):
    raise TypeError(
        "Expected list for 'dilations' argument to "
        "'conv3d_backprop_filter_v2' Op, not %r." % dilations)
  dilations = [_execute.make_int(_i, "dilations") for _i in dilations]
  _attr_T, _inputs_T = _execute.args_to_matching_eager([input, out_backprop], ctx, [_dtypes.half, _dtypes.bfloat16, _dtypes.float32, _dtypes.float64, ])
  (input, out_backprop) = _inputs_T
  filter_sizes = _ops.convert_to_tensor(filter_sizes, _dtypes.int32)
  _inputs_flat = [input, filter_sizes, out_backprop]
  _attrs = ("T", _attr_T, "strides", strides, "padding", padding,
  "data_format", data_format, "dilations", dilations)
  _result = _execute.execute(b"Conv3DBackpropFilterV2", 1,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "Conv3DBackpropFilterV2", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_Conv3DBackpropInput_T = TypeVar("TV_Conv3DBackpropInput_T", _atypes.Float32, _atypes.Float64, _atypes.Half)

def conv3d_backprop_input(input: Annotated[Any, TV_Conv3DBackpropInput_T], filter: Annotated[Any, TV_Conv3DBackpropInput_T], out_backprop: Annotated[Any, TV_Conv3DBackpropInput_T], strides, padding: str, dilations=[1, 1, 1, 1, 1], name=None) -> Annotated[Any, TV_Conv3DBackpropInput_T]:
  r"""Computes the gradients of 3-D convolution with respect to the input.

  Args:
    input: A `Tensor`. Must be one of the following types: `half`, `float32`, `float64`.
      Shape `[batch, depth, rows, cols, in_channels]`.
    filter: A `Tensor`. Must have the same type as `input`.
      Shape `[depth, rows, cols, in_channels, out_channels]`.
      `in_channels` must match between `input` and `filter`.
    out_backprop: A `Tensor`. Must have the same type as `input`.
      Backprop signal of shape `[batch, out_depth, out_rows, out_cols,
      out_channels]`.
    strides: A list of `ints` that has length `>= 5`.
      1-D tensor of length 5. The stride of the sliding window for each
      dimension of `input`. Must have `strides[0] = strides[4] = 1`.
    padding: A `string` from: `"SAME", "VALID"`.
      The type of padding algorithm to use.
    dilations: An optional list of `ints`. Defaults to `[1, 1, 1, 1, 1]`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "Conv3DBackpropInput", name, input, filter, out_backprop,
        "strides", strides, "padding", padding, "dilations", dilations)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return conv3d_backprop_input_eager_fallback(
          input, filter, out_backprop, strides=strides, padding=padding,
          dilations=dilations, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if not isinstance(strides, (list, tuple)):
    raise TypeError(
        "Expected list for 'strides' argument to "
        "'conv3d_backprop_input' Op, not %r." % strides)
  strides = [_execute.make_int(_i, "strides") for _i in strides]
  padding = _execute.make_str(padding, "padding")
  if dilations is None:
    dilations = [1, 1, 1, 1, 1]
  if not isinstance(dilations, (list, tuple)):
    raise TypeError(
        "Expected list for 'dilations' argument to "
        "'conv3d_backprop_input' Op, not %r." % dilations)
  dilations = [_execute.make_int(_i, "dilations") for _i in dilations]
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "Conv3DBackpropInput", input=input, filter=filter,
                               out_backprop=out_backprop, strides=strides,
                               padding=padding, dilations=dilations,
                               name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"), "strides",
              _op.get_attr("strides"), "padding", _op.get_attr("padding"),
              "dilations", _op.get_attr("dilations"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "Conv3DBackpropInput", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

Conv3DBackpropInput = tf_export("raw_ops.Conv3DBackpropInput")(_ops.to_raw_op(conv3d_backprop_input))


def conv3d_backprop_input_eager_fallback(input: Annotated[Any, TV_Conv3DBackpropInput_T], filter: Annotated[Any, TV_Conv3DBackpropInput_T], out_backprop: Annotated[Any, TV_Conv3DBackpropInput_T], strides, padding: str, dilations, name, ctx) -> Annotated[Any, TV_Conv3DBackpropInput_T]:
  if not isinstance(strides, (list, tuple)):
    raise TypeError(
        "Expected list for 'strides' argument to "
        "'conv3d_backprop_input' Op, not %r." % strides)
  strides = [_execute.make_int(_i, "strides") for _i in strides]
  padding = _execute.make_str(padding, "padding")
  if dilations is None:
    dilations = [1, 1, 1, 1, 1]
  if not isinstance(dilations, (list, tuple)):
    raise TypeError(
        "Expected list for 'dilations' argument to "
        "'conv3d_backprop_input' Op, not %r." % dilations)
  dilations = [_execute.make_int(_i, "dilations") for _i in dilations]
  _attr_T, _inputs_T = _execute.args_to_matching_eager([input, filter, out_backprop], ctx, [_dtypes.half, _dtypes.float32, _dtypes.float64, ])
  (input, filter, out_backprop) = _inputs_T
  _inputs_flat = [input, filter, out_backprop]
  _attrs = ("T", _attr_T, "strides", strides, "padding", padding, "dilations",
  dilations)
  _result = _execute.execute(b"Conv3DBackpropInput", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "Conv3DBackpropInput", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_Conv3DBackpropInputV2_T = TypeVar("TV_Conv3DBackpropInputV2_T", _atypes.BFloat16, _atypes.Float32, _atypes.Float64, _atypes.Half)
TV_Conv3DBackpropInputV2_Tshape = TypeVar("TV_Conv3DBackpropInputV2_Tshape", _atypes.Int32, _atypes.Int64)

def conv3d_backprop_input_v2(input_sizes: Annotated[Any, TV_Conv3DBackpropInputV2_Tshape], filter: Annotated[Any, TV_Conv3DBackpropInputV2_T], out_backprop: Annotated[Any, TV_Conv3DBackpropInputV2_T], strides, padding: str, data_format:str="NDHWC", dilations=[1, 1, 1, 1, 1], name=None) -> Annotated[Any, TV_Conv3DBackpropInputV2_T]:
  r"""Computes the gradients of 3-D convolution with respect to the input.

  Args:
    input_sizes: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      An integer vector representing the tensor shape of `input`,
      where `input` is a 5-D
      `[batch, depth, rows, cols, in_channels]` tensor.
    filter: A `Tensor`. Must be one of the following types: `half`, `bfloat16`, `float32`, `float64`.
      Shape `[depth, rows, cols, in_channels, out_channels]`.
      `in_channels` must match between `input` and `filter`.
    out_backprop: A `Tensor`. Must have the same type as `filter`.
      Backprop signal of shape `[batch, out_depth, out_rows, out_cols,
      out_channels]`.
    strides: A list of `ints` that has length `>= 5`.
      1-D tensor of length 5. The stride of the sliding window for each
      dimension of `input`. Must have `strides[0] = strides[4] = 1`.
    padding: A `string` from: `"SAME", "VALID"`.
      The type of padding algorithm to use.
    data_format: An optional `string` from: `"NDHWC", "NCDHW"`. Defaults to `"NDHWC"`.
      The data format of the input and output data. With the
      default format "NDHWC", the data is stored in the order of:
          [batch, in_depth, in_height, in_width, in_channels].
      Alternatively, the format could be "NCDHW", the data storage order is:
          [batch, in_channels, in_depth, in_height, in_width].
    dilations: An optional list of `ints`. Defaults to `[1, 1, 1, 1, 1]`.
      1-D tensor of length 5.  The dilation factor for each dimension of
      `input`. If set to k > 1, there will be k-1 skipped cells between each
      filter element on that dimension. The dimension order is determined by the
      value of `data_format`, see above for details. Dilations in the batch and
      depth dimensions must be 1.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `filter`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "Conv3DBackpropInputV2", name, input_sizes, filter,
        out_backprop, "strides", strides, "padding", padding, "data_format",
        data_format, "dilations", dilations)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return conv3d_backprop_input_v2_eager_fallback(
          input_sizes, filter, out_backprop, strides=strides, padding=padding,
          data_format=data_format, dilations=dilations, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if not isinstance(strides, (list, tuple)):
    raise TypeError(
        "Expected list for 'strides' argument to "
        "'conv3d_backprop_input_v2' Op, not %r." % strides)
  strides = [_execute.make_int(_i, "strides") for _i in strides]
  padding = _execute.make_str(padding, "padding")
  if data_format is None:
    data_format = "NDHWC"
  data_format = _execute.make_str(data_format, "data_format")
  if dilations is None:
    dilations = [1, 1, 1, 1, 1]
  if not isinstance(dilations, (list, tuple)):
    raise TypeError(
        "Expected list for 'dilations' argument to "
        "'conv3d_backprop_input_v2' Op, not %r." % dilations)
  dilations = [_execute.make_int(_i, "dilations") for _i in dilations]
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "Conv3DBackpropInputV2", input_sizes=input_sizes, filter=filter,
                                 out_backprop=out_backprop, strides=strides,
                                 padding=padding, data_format=data_format,
                                 dilations=dilations, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"), "strides",
              _op.get_attr("strides"), "padding", _op.get_attr("padding"),
              "data_format", _op.get_attr("data_format"), "dilations",
              _op.get_attr("dilations"), "Tshape",
              _op._get_attr_type("Tshape"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "Conv3DBackpropInputV2", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

Conv3DBackpropInputV2 = tf_export("raw_ops.Conv3DBackpropInputV2")(_ops.to_raw_op(conv3d_backprop_input_v2))


def conv3d_backprop_input_v2_eager_fallback(input_sizes: Annotated[Any, TV_Conv3DBackpropInputV2_Tshape], filter: Annotated[Any, TV_Conv3DBackpropInputV2_T], out_backprop: Annotated[Any, TV_Conv3DBackpropInputV2_T], strides, padding: str, data_format: str, dilations, name, ctx) -> Annotated[Any, TV_Conv3DBackpropInputV2_T]:
  if not isinstance(strides, (list, tuple)):
    raise TypeError(
        "Expected list for 'strides' argument to "
        "'conv3d_backprop_input_v2' Op, not %r." % strides)
  strides = [_execute.make_int(_i, "strides") for _i in strides]
  padding = _execute.make_str(padding, "padding")
  if data_format is None:
    data_format = "NDHWC"
  data_format = _execute.make_str(data_format, "data_format")
  if dilations is None:
    dilations = [1, 1, 1, 1, 1]
  if not isinstance(dilations, (list, tuple)):
    raise TypeError(
        "Expected list for 'dilations' argument to "
        "'conv3d_backprop_input_v2' Op, not %r." % dilations)
  dilations = [_execute.make_int(_i, "dilations") for _i in dilations]
  _attr_T, _inputs_T = _execute.args_to_matching_eager([filter, out_backprop], ctx, [_dtypes.half, _dtypes.bfloat16, _dtypes.float32, _dtypes.float64, ])
  (filter, out_backprop) = _inputs_T
  _attr_Tshape, (input_sizes,) = _execute.args_to_matching_eager([input_sizes], ctx, [_dtypes.int32, _dtypes.int64, ], _dtypes.int32)
  _inputs_flat = [input_sizes, filter, out_backprop]
  _attrs = ("T", _attr_T, "strides", strides, "padding", padding,
  "data_format", data_format, "dilations", dilations, "Tshape", _attr_Tshape)
  _result = _execute.execute(b"Conv3DBackpropInputV2", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "Conv3DBackpropInputV2", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_DataFormatDimMap_T = TypeVar("TV_DataFormatDimMap_T", _atypes.Int32, _atypes.Int64)

def data_format_dim_map(x: Annotated[Any, TV_DataFormatDimMap_T], src_format:str="NHWC", dst_format:str="NCHW", name=None) -> Annotated[Any, TV_DataFormatDimMap_T]:
  r"""Returns the dimension index in the destination data format given the one in

  the source data format.

  Args:
    x: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      A Tensor with each element as a dimension index in source data format.
      Must be in the range [-4, 4).
    src_format: An optional `string`. Defaults to `"NHWC"`.
      source data format.
    dst_format: An optional `string`. Defaults to `"NCHW"`.
      destination data format.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `x`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "DataFormatDimMap", name, x, "src_format", src_format,
        "dst_format", dst_format)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return data_format_dim_map_eager_fallback(
          x, src_format=src_format, dst_format=dst_format, name=name,
          ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if src_format is None:
    src_format = "NHWC"
  src_format = _execute.make_str(src_format, "src_format")
  if dst_format is None:
    dst_format = "NCHW"
  dst_format = _execute.make_str(dst_format, "dst_format")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "DataFormatDimMap", x=x, src_format=src_format, dst_format=dst_format,
                            name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"), "src_format",
              _op.get_attr("src_format"), "dst_format",
              _op.get_attr("dst_format"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "DataFormatDimMap", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

DataFormatDimMap = tf_export("raw_ops.DataFormatDimMap")(_ops.to_raw_op(data_format_dim_map))


def data_format_dim_map_eager_fallback(x: Annotated[Any, TV_DataFormatDimMap_T], src_format: str, dst_format: str, name, ctx) -> Annotated[Any, TV_DataFormatDimMap_T]:
  if src_format is None:
    src_format = "NHWC"
  src_format = _execute.make_str(src_format, "src_format")
  if dst_format is None:
    dst_format = "NCHW"
  dst_format = _execute.make_str(dst_format, "dst_format")
  _attr_T, (x,) = _execute.args_to_matching_eager([x], ctx, [_dtypes.int32, _dtypes.int64, ], _dtypes.int32)
  _inputs_flat = [x]
  _attrs = ("T", _attr_T, "src_format", src_format, "dst_format", dst_format)
  _result = _execute.execute(b"DataFormatDimMap", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "DataFormatDimMap", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_DataFormatVecPermute_T = TypeVar("TV_DataFormatVecPermute_T", _atypes.Int32, _atypes.Int64)

def data_format_vec_permute(x: Annotated[Any, TV_DataFormatVecPermute_T], src_format:str="NHWC", dst_format:str="NCHW", name=None) -> Annotated[Any, TV_DataFormatVecPermute_T]:
  r"""Permute input tensor from `src_format` to `dst_format`.

  Given source and destination format strings of length n=4 or 5, the input
  tensor must be a vector of size n or n-2, or a 2D tensor of shape
  (n, 2) or (n-2, 2).

  If the first dimension of the input tensor is n-2, it is assumed that
  non-spatial dimensions are omitted (i.e `N`, `C`).

  For example, with `src_format` of `NHWC`, `dst_format` of `NCHW`, and input:
  ```
  [1, 2, 3, 4]
  ```
  , the output will be:
  ```
  [1, 4, 2, 3]
  ```
  With `src_format` of `NDHWC`, `dst_format` of `NCDHW`, and input:
  ```
  [[1, 6], [2, 7], [3, 8], [4, 9], [5, 10]]
  ```
  , the output will be:
  ```
  [[1, 6], [5, 10], [2, 7], [3, 8], [4, 9]]
  ```
  With `src_format` of `NHWC`, `dst_format` of `NCHW`, and input:
  ```
  [1, 2]
  ```
  , the output will be:
  ```
  [1, 2]
  ```

  Args:
    x: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      Tensor of rank 1 or 2 in source data format.
    src_format: An optional `string`. Defaults to `"NHWC"`.
      source data format.
    dst_format: An optional `string`. Defaults to `"NCHW"`.
      destination data format.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `x`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "DataFormatVecPermute", name, x, "src_format", src_format,
        "dst_format", dst_format)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return data_format_vec_permute_eager_fallback(
          x, src_format=src_format, dst_format=dst_format, name=name,
          ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if src_format is None:
    src_format = "NHWC"
  src_format = _execute.make_str(src_format, "src_format")
  if dst_format is None:
    dst_format = "NCHW"
  dst_format = _execute.make_str(dst_format, "dst_format")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "DataFormatVecPermute", x=x, src_format=src_format,
                                dst_format=dst_format, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"), "src_format",
              _op.get_attr("src_format"), "dst_format",
              _op.get_attr("dst_format"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "DataFormatVecPermute", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

DataFormatVecPermute = tf_export("raw_ops.DataFormatVecPermute")(_ops.to_raw_op(data_format_vec_permute))


def data_format_vec_permute_eager_fallback(x: Annotated[Any, TV_DataFormatVecPermute_T], src_format: str, dst_format: str, name, ctx) -> Annotated[Any, TV_DataFormatVecPermute_T]:
  if src_format is None:
    src_format = "NHWC"
  src_format = _execute.make_str(src_format, "src_format")
  if dst_format is None:
    dst_format = "NCHW"
  dst_format = _execute.make_str(dst_format, "dst_format")
  _attr_T, (x,) = _execute.args_to_matching_eager([x], ctx, [_dtypes.int32, _dtypes.int64, ], _dtypes.int32)
  _inputs_flat = [x]
  _attrs = ("T", _attr_T, "src_format", src_format, "dst_format", dst_format)
  _result = _execute.execute(b"DataFormatVecPermute", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "DataFormatVecPermute", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_DepthwiseConv2dNative_T = TypeVar("TV_DepthwiseConv2dNative_T", _atypes.BFloat16, _atypes.Float32, _atypes.Float64, _atypes.Half)

def depthwise_conv2d_native(input: Annotated[Any, TV_DepthwiseConv2dNative_T], filter: Annotated[Any, TV_DepthwiseConv2dNative_T], strides, padding: str, explicit_paddings=[], data_format:str="NHWC", dilations=[1, 1, 1, 1], name=None) -> Annotated[Any, TV_DepthwiseConv2dNative_T]:
  r"""Computes a 2-D depthwise convolution given 4-D `input` and `filter` tensors.

  Given an input tensor of shape `[batch, in_height, in_width, in_channels]`
  and a filter / kernel tensor of shape
  `[filter_height, filter_width, in_channels, channel_multiplier]`, containing
  `in_channels` convolutional filters of depth 1, `depthwise_conv2d` applies
  a different filter to each input channel (expanding from 1 channel to
  `channel_multiplier` channels for each), then concatenates the results
  together. Thus, the output has `in_channels * channel_multiplier` channels.

  ```
  for k in 0..in_channels-1
    for q in 0..channel_multiplier-1
      output[b, i, j, k * channel_multiplier + q] =
        sum_{di, dj} input[b, strides[1] * i + di, strides[2] * j + dj, k] *
                          filter[di, dj, k, q]
  ```

  Must have `strides[0] = strides[3] = 1`.  For the most common case of the same
  horizontal and vertices strides, `strides = [1, stride, stride, 1]`.

  Args:
    input: A `Tensor`. Must be one of the following types: `half`, `bfloat16`, `float32`, `float64`.
    filter: A `Tensor`. Must have the same type as `input`.
    strides: A list of `ints`.
      1-D of length 4.  The stride of the sliding window for each dimension
      of `input`.
    padding: A `string` from: `"SAME", "VALID", "EXPLICIT"`.
      The type of padding algorithm to use.
    explicit_paddings: An optional list of `ints`. Defaults to `[]`.
    data_format: An optional `string` from: `"NHWC", "NCHW"`. Defaults to `"NHWC"`.
      Specify the data format of the input and output data. With the
      default format "NHWC", the data is stored in the order of:
          [batch, height, width, channels].
      Alternatively, the format could be "NCHW", the data storage order of:
          [batch, channels, height, width].
    dilations: An optional list of `ints`. Defaults to `[1, 1, 1, 1]`.
      1-D tensor of length 4.  The dilation factor for each dimension of
      `input`. If set to k > 1, there will be k-1 skipped cells between each filter
      element on that dimension. The dimension order is determined by the value of
      `data_format`, see above for details. Dilations in the batch and depth
      dimensions must be 1.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "DepthwiseConv2dNative", name, input, filter, "strides",
        strides, "padding", padding, "explicit_paddings", explicit_paddings,
        "data_format", data_format, "dilations", dilations)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return depthwise_conv2d_native_eager_fallback(
          input, filter, strides=strides, padding=padding,
          explicit_paddings=explicit_paddings, data_format=data_format,
          dilations=dilations, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if not isinstance(strides, (list, tuple)):
    raise TypeError(
        "Expected list for 'strides' argument to "
        "'depthwise_conv2d_native' Op, not %r." % strides)
  strides = [_execute.make_int(_i, "strides") for _i in strides]
  padding = _execute.make_str(padding, "padding")
  if explicit_paddings is None:
    explicit_paddings = []
  if not isinstance(explicit_paddings, (list, tuple)):
    raise TypeError(
        "Expected list for 'explicit_paddings' argument to "
        "'depthwise_conv2d_native' Op, not %r." % explicit_paddings)
  explicit_paddings = [_execute.make_int(_i, "explicit_paddings") for _i in explicit_paddings]
  if data_format is None:
    data_format = "NHWC"
  data_format = _execute.make_str(data_format, "data_format")
  if dilations is None:
    dilations = [1, 1, 1, 1]
  if not isinstance(dilations, (list, tuple)):
    raise TypeError(
        "Expected list for 'dilations' argument to "
        "'depthwise_conv2d_native' Op, not %r." % dilations)
  dilations = [_execute.make_int(_i, "dilations") for _i in dilations]
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "DepthwiseConv2dNative", input=input, filter=filter, strides=strides,
                                 padding=padding,
                                 explicit_paddings=explicit_paddings,
                                 data_format=data_format, dilations=dilations,
                                 name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"), "strides",
              _op.get_attr("strides"), "padding", _op.get_attr("padding"),
              "explicit_paddings", _op.get_attr("explicit_paddings"),
              "data_format", _op.get_attr("data_format"), "dilations",
              _op.get_attr("dilations"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "DepthwiseConv2dNative", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

DepthwiseConv2dNative = tf_export("raw_ops.DepthwiseConv2dNative")(_ops.to_raw_op(depthwise_conv2d_native))


def depthwise_conv2d_native_eager_fallback(input: Annotated[Any, TV_DepthwiseConv2dNative_T], filter: Annotated[Any, TV_DepthwiseConv2dNative_T], strides, padding: str, explicit_paddings, data_format: str, dilations, name, ctx) -> Annotated[Any, TV_DepthwiseConv2dNative_T]:
  if not isinstance(strides, (list, tuple)):
    raise TypeError(
        "Expected list for 'strides' argument to "
        "'depthwise_conv2d_native' Op, not %r." % strides)
  strides = [_execute.make_int(_i, "strides") for _i in strides]
  padding = _execute.make_str(padding, "padding")
  if explicit_paddings is None:
    explicit_paddings = []
  if not isinstance(explicit_paddings, (list, tuple)):
    raise TypeError(
        "Expected list for 'explicit_paddings' argument to "
        "'depthwise_conv2d_native' Op, not %r." % explicit_paddings)
  explicit_paddings = [_execute.make_int(_i, "explicit_paddings") for _i in explicit_paddings]
  if data_format is None:
    data_format = "NHWC"
  data_format = _execute.make_str(data_format, "data_format")
  if dilations is None:
    dilations = [1, 1, 1, 1]
  if not isinstance(dilations, (list, tuple)):
    raise TypeError(
        "Expected list for 'dilations' argument to "
        "'depthwise_conv2d_native' Op, not %r." % dilations)
  dilations = [_execute.make_int(_i, "dilations") for _i in dilations]
  _attr_T, _inputs_T = _execute.args_to_matching_eager([input, filter], ctx, [_dtypes.half, _dtypes.bfloat16, _dtypes.float32, _dtypes.float64, ])
  (input, filter) = _inputs_T
  _inputs_flat = [input, filter]
  _attrs = ("T", _attr_T, "strides", strides, "padding", padding,
  "explicit_paddings", explicit_paddings, "data_format", data_format,
  "dilations", dilations)
  _result = _execute.execute(b"DepthwiseConv2dNative", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "DepthwiseConv2dNative", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_DepthwiseConv2dNativeBackpropFilter_T = TypeVar("TV_DepthwiseConv2dNativeBackpropFilter_T", _atypes.BFloat16, _atypes.Float32, _atypes.Float64, _atypes.Half)

def depthwise_conv2d_native_backprop_filter(input: Annotated[Any, TV_DepthwiseConv2dNativeBackpropFilter_T], filter_sizes: Annotated[Any, _atypes.Int32], out_backprop: Annotated[Any, TV_DepthwiseConv2dNativeBackpropFilter_T], strides, padding: str, explicit_paddings=[], data_format:str="NHWC", dilations=[1, 1, 1, 1], name=None) -> Annotated[Any, TV_DepthwiseConv2dNativeBackpropFilter_T]:
  r"""Computes the gradients of depthwise convolution with respect to the filter.

  Args:
    input: A `Tensor`. Must be one of the following types: `half`, `bfloat16`, `float32`, `float64`.
      4-D with shape based on `data_format`.  For example, if
      `data_format` is 'NHWC' then `input` is a 4-D `[batch, in_height,
      in_width, in_channels]` tensor.
    filter_sizes: A `Tensor` of type `int32`.
      An integer vector representing the tensor shape of `filter`,
      where `filter` is a 4-D
      `[filter_height, filter_width, in_channels, depthwise_multiplier]` tensor.
    out_backprop: A `Tensor`. Must have the same type as `input`.
      4-D with shape  based on `data_format`.
      For example, if `data_format` is 'NHWC' then
      out_backprop shape is `[batch, out_height, out_width, out_channels]`.
      Gradients w.r.t. the output of the convolution.
    strides: A list of `ints`.
      The stride of the sliding window for each dimension of the input
      of the convolution.
    padding: A `string` from: `"SAME", "VALID", "EXPLICIT"`.
      The type of padding algorithm to use.
    explicit_paddings: An optional list of `ints`. Defaults to `[]`.
    data_format: An optional `string` from: `"NHWC", "NCHW"`. Defaults to `"NHWC"`.
      Specify the data format of the input and output data. With the
      default format "NHWC", the data is stored in the order of:
          [batch, height, width, channels].
      Alternatively, the format could be "NCHW", the data storage order of:
          [batch, channels, height, width].
    dilations: An optional list of `ints`. Defaults to `[1, 1, 1, 1]`.
      1-D tensor of length 4.  The dilation factor for each dimension of
      `input`. If set to k > 1, there will be k-1 skipped cells between each filter
      element on that dimension. The dimension order is determined by the value of
      `data_format`, see above for details. Dilations in the batch and depth
      dimensions must be 1.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "DepthwiseConv2dNativeBackpropFilter", name, input,
        filter_sizes, out_backprop, "strides", strides, "padding", padding,
        "explicit_paddings", explicit_paddings, "data_format", data_format,
        "dilations", dilations)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return depthwise_conv2d_native_backprop_filter_eager_fallback(
          input, filter_sizes, out_backprop, strides=strides, padding=padding,
          explicit_paddings=explicit_paddings, data_format=data_format,
          dilations=dilations, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if not isinstance(strides, (list, tuple)):
    raise TypeError(
        "Expected list for 'strides' argument to "
        "'depthwise_conv2d_native_backprop_filter' Op, not %r." % strides)
  strides = [_execute.make_int(_i, "strides") for _i in strides]
  padding = _execute.make_str(padding, "padding")
  if explicit_paddings is None:
    explicit_paddings = []
  if not isinstance(explicit_paddings, (list, tuple)):
    raise TypeError(
        "Expected list for 'explicit_paddings' argument to "
        "'depthwise_conv2d_native_backprop_filter' Op, not %r." % explicit_paddings)
  explicit_paddings = [_execute.make_int(_i, "explicit_paddings") for _i in explicit_paddings]
  if data_format is None:
    data_format = "NHWC"
  data_format = _execute.make_str(data_format, "data_format")
  if dilations is None:
    dilations = [1, 1, 1, 1]
  if not isinstance(dilations, (list, tuple)):
    raise TypeError(
        "Expected list for 'dilations' argument to "
        "'depthwise_conv2d_native_backprop_filter' Op, not %r." % dilations)
  dilations = [_execute.make_int(_i, "dilations") for _i in dilations]
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "DepthwiseConv2dNativeBackpropFilter", input=input,
                                               filter_sizes=filter_sizes,
                                               out_backprop=out_backprop,
                                               strides=strides,
                                               padding=padding,
                                               explicit_paddings=explicit_paddings,
                                               data_format=data_format,
                                               dilations=dilations, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"), "strides",
              _op.get_attr("strides"), "padding", _op.get_attr("padding"),
              "explicit_paddings", _op.get_attr("explicit_paddings"),
              "data_format", _op.get_attr("data_format"), "dilations",
              _op.get_attr("dilations"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "DepthwiseConv2dNativeBackpropFilter", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

DepthwiseConv2dNativeBackpropFilter = tf_export("raw_ops.DepthwiseConv2dNativeBackpropFilter")(_ops.to_raw_op(depthwise_conv2d_native_backprop_filter))


def depthwise_conv2d_native_backprop_filter_eager_fallback(input: Annotated[Any, TV_DepthwiseConv2dNativeBackpropFilter_T], filter_sizes: Annotated[Any, _atypes.Int32], out_backprop: Annotated[Any, TV_DepthwiseConv2dNativeBackpropFilter_T], strides, padding: str, explicit_paddings, data_format: str, dilations, name, ctx) -> Annotated[Any, TV_DepthwiseConv2dNativeBackpropFilter_T]:
  if not isinstance(strides, (list, tuple)):
    raise TypeError(
        "Expected list for 'strides' argument to "
        "'depthwise_conv2d_native_backprop_filter' Op, not %r." % strides)
  strides = [_execute.make_int(_i, "strides") for _i in strides]
  padding = _execute.make_str(padding, "padding")
  if explicit_paddings is None:
    explicit_paddings = []
  if not isinstance(explicit_paddings, (list, tuple)):
    raise TypeError(
        "Expected list for 'explicit_paddings' argument to "
        "'depthwise_conv2d_native_backprop_filter' Op, not %r." % explicit_paddings)
  explicit_paddings = [_execute.make_int(_i, "explicit_paddings") for _i in explicit_paddings]
  if data_format is None:
    data_format = "NHWC"
  data_format = _execute.make_str(data_format, "data_format")
  if dilations is None:
    dilations = [1, 1, 1, 1]
  if not isinstance(dilations, (list, tuple)):
    raise TypeError(
        "Expected list for 'dilations' argument to "
        "'depthwise_conv2d_native_backprop_filter' Op, not %r." % dilations)
  dilations = [_execute.make_int(_i, "dilations") for _i in dilations]
  _attr_T, _inputs_T = _execute.args_to_matching_eager([input, out_backprop], ctx, [_dtypes.half, _dtypes.bfloat16, _dtypes.float32, _dtypes.float64, ])
  (input, out_backprop) = _inputs_T
  filter_sizes = _ops.convert_to_tensor(filter_sizes, _dtypes.int32)
  _inputs_flat = [input, filter_sizes, out_backprop]
  _attrs = ("T", _attr_T, "strides", strides, "padding", padding,
  "explicit_paddings", explicit_paddings, "data_format", data_format,
  "dilations", dilations)
  _result = _execute.execute(b"DepthwiseConv2dNativeBackpropFilter", 1,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "DepthwiseConv2dNativeBackpropFilter", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_DepthwiseConv2dNativeBackpropInput_T = TypeVar("TV_DepthwiseConv2dNativeBackpropInput_T", _atypes.BFloat16, _atypes.Float32, _atypes.Float64, _atypes.Half)

def depthwise_conv2d_native_backprop_input(input_sizes: Annotated[Any, _atypes.Int32], filter: Annotated[Any, TV_DepthwiseConv2dNativeBackpropInput_T], out_backprop: Annotated[Any, TV_DepthwiseConv2dNativeBackpropInput_T], strides, padding: str, explicit_paddings=[], data_format:str="NHWC", dilations=[1, 1, 1, 1], name=None) -> Annotated[Any, TV_DepthwiseConv2dNativeBackpropInput_T]:
  r"""Computes the gradients of depthwise convolution with respect to the input.

  Args:
    input_sizes: A `Tensor` of type `int32`.
      An integer vector representing the shape of `input`, based
      on `data_format`.  For example, if `data_format` is 'NHWC' then
       `input` is a 4-D `[batch, height, width, channels]` tensor.
    filter: A `Tensor`. Must be one of the following types: `half`, `bfloat16`, `float32`, `float64`.
      4-D with shape
      `[filter_height, filter_width, in_channels, depthwise_multiplier]`.
    out_backprop: A `Tensor`. Must have the same type as `filter`.
      4-D with shape  based on `data_format`.
      For example, if `data_format` is 'NHWC' then
      out_backprop shape is `[batch, out_height, out_width, out_channels]`.
      Gradients w.r.t. the output of the convolution.
    strides: A list of `ints`.
      The stride of the sliding window for each dimension of the input
      of the convolution.
    padding: A `string` from: `"SAME", "VALID", "EXPLICIT"`.
      The type of padding algorithm to use.
    explicit_paddings: An optional list of `ints`. Defaults to `[]`.
    data_format: An optional `string` from: `"NHWC", "NCHW"`. Defaults to `"NHWC"`.
      Specify the data format of the input and output data. With the
      default format "NHWC", the data is stored in the order of:
          [batch, height, width, channels].
      Alternatively, the format could be "NCHW", the data storage order of:
          [batch, channels, height, width].
    dilations: An optional list of `ints`. Defaults to `[1, 1, 1, 1]`.
      1-D tensor of length 4.  The dilation factor for each dimension of
      `input`. If set to k > 1, there will be k-1 skipped cells between each filter
      element on that dimension. The dimension order is determined by the value of
      `data_format`, see above for details. Dilations in the batch and depth
      dimensions must be 1.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `filter`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "DepthwiseConv2dNativeBackpropInput", name, input_sizes, filter,
        out_backprop, "strides", strides, "padding", padding,
        "explicit_paddings", explicit_paddings, "data_format", data_format,
        "dilations", dilations)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return depthwise_conv2d_native_backprop_input_eager_fallback(
          input_sizes, filter, out_backprop, strides=strides, padding=padding,
          explicit_paddings=explicit_paddings, data_format=data_format,
          dilations=dilations, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if not isinstance(strides, (list, tuple)):
    raise TypeError(
        "Expected list for 'strides' argument to "
        "'depthwise_conv2d_native_backprop_input' Op, not %r." % strides)
  strides = [_execute.make_int(_i, "strides") for _i in strides]
  padding = _execute.make_str(padding, "padding")
  if explicit_paddings is None:
    explicit_paddings = []
  if not isinstance(explicit_paddings, (list, tuple)):
    raise TypeError(
        "Expected list for 'explicit_paddings' argument to "
        "'depthwise_conv2d_native_backprop_input' Op, not %r." % explicit_paddings)
  explicit_paddings = [_execute.make_int(_i, "explicit_paddings") for _i in explicit_paddings]
  if data_format is None:
    data_format = "NHWC"
  data_format = _execute.make_str(data_format, "data_format")
  if dilations is None:
    dilations = [1, 1, 1, 1]
  if not isinstance(dilations, (list, tuple)):
    raise TypeError(
        "Expected list for 'dilations' argument to "
        "'depthwise_conv2d_native_backprop_input' Op, not %r." % dilations)
  dilations = [_execute.make_int(_i, "dilations") for _i in dilations]
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "DepthwiseConv2dNativeBackpropInput", input_sizes=input_sizes,
                                              filter=filter,
                                              out_backprop=out_backprop,
                                              strides=strides,
                                              padding=padding,
                                              explicit_paddings=explicit_paddings,
                                              data_format=data_format,
                                              dilations=dilations, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"), "strides",
              _op.get_attr("strides"), "padding", _op.get_attr("padding"),
              "explicit_paddings", _op.get_attr("explicit_paddings"),
              "data_format", _op.get_attr("data_format"), "dilations",
              _op.get_attr("dilations"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "DepthwiseConv2dNativeBackpropInput", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

DepthwiseConv2dNativeBackpropInput = tf_export("raw_ops.DepthwiseConv2dNativeBackpropInput")(_ops.to_raw_op(depthwise_conv2d_native_backprop_input))


def depthwise_conv2d_native_backprop_input_eager_fallback(input_sizes: Annotated[Any, _atypes.Int32], filter: Annotated[Any, TV_DepthwiseConv2dNativeBackpropInput_T], out_backprop: Annotated[Any, TV_DepthwiseConv2dNativeBackpropInput_T], strides, padding: str, explicit_paddings, data_format: str, dilations, name, ctx) -> Annotated[Any, TV_DepthwiseConv2dNativeBackpropInput_T]:
  if not isinstance(strides, (list, tuple)):
    raise TypeError(
        "Expected list for 'strides' argument to "
        "'depthwise_conv2d_native_backprop_input' Op, not %r." % strides)
  strides = [_execute.make_int(_i, "strides") for _i in strides]
  padding = _execute.make_str(padding, "padding")
  if explicit_paddings is None:
    explicit_paddings = []
  if not isinstance(explicit_paddings, (list, tuple)):
    raise TypeError(
        "Expected list for 'explicit_paddings' argument to "
        "'depthwise_conv2d_native_backprop_input' Op, not %r." % explicit_paddings)
  explicit_paddings = [_execute.make_int(_i, "explicit_paddings") for _i in explicit_paddings]
  if data_format is None:
    data_format = "NHWC"
  data_format = _execute.make_str(data_format, "data_format")
  if dilations is None:
    dilations = [1, 1, 1, 1]
  if not isinstance(dilations, (list, tuple)):
    raise TypeError(
        "Expected list for 'dilations' argument to "
        "'depthwise_conv2d_native_backprop_input' Op, not %r." % dilations)
  dilations = [_execute.make_int(_i, "dilations") for _i in dilations]
  _attr_T, _inputs_T = _execute.args_to_matching_eager([filter, out_backprop], ctx, [_dtypes.half, _dtypes.bfloat16, _dtypes.float32, _dtypes.float64, ])
  (filter, out_backprop) = _inputs_T
  input_sizes = _ops.convert_to_tensor(input_sizes, _dtypes.int32)
  _inputs_flat = [input_sizes, filter, out_backprop]
  _attrs = ("T", _attr_T, "strides", strides, "padding", padding,
  "explicit_paddings", explicit_paddings, "data_format", data_format,
  "dilations", dilations)
  _result = _execute.execute(b"DepthwiseConv2dNativeBackpropInput", 1,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "DepthwiseConv2dNativeBackpropInput", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_Dilation2D_T = TypeVar("TV_Dilation2D_T", _atypes.BFloat16, _atypes.Float32, _atypes.Float64, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int64, _atypes.Int8, _atypes.UInt16, _atypes.UInt32, _atypes.UInt64, _atypes.UInt8)

def dilation2d(input: Annotated[Any, TV_Dilation2D_T], filter: Annotated[Any, TV_Dilation2D_T], strides, rates, padding: str, name=None) -> Annotated[Any, TV_Dilation2D_T]:
  r"""Computes the grayscale dilation of 4-D `input` and 3-D `filter` tensors.

  The `input` tensor has shape `[batch, in_height, in_width, depth]` and the
  `filter` tensor has shape `[filter_height, filter_width, depth]`, i.e., each
  input channel is processed independently of the others with its own structuring
  function. The `output` tensor has shape
  `[batch, out_height, out_width, depth]`. The spatial dimensions of the output
  tensor depend on the `padding` algorithm. We currently only support the default
  "NHWC" `data_format`.

  In detail, the grayscale morphological 2-D dilation is the max-sum correlation
  (for consistency with `conv2d`, we use unmirrored filters):

      output[b, y, x, c] =
         max_{dy, dx} input[b,
                            strides[1] * y + rates[1] * dy,
                            strides[2] * x + rates[2] * dx,
                            c] +
                      filter[dy, dx, c]

  Max-pooling is a special case when the filter has size equal to the pooling
  kernel size and contains all zeros.

  Note on duality: The dilation of `input` by the `filter` is equal to the
  negation of the erosion of `-input` by the reflected `filter`.

  Args:
    input: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`, `int64`, `bfloat16`, `uint16`, `half`, `uint32`, `uint64`.
      4-D with shape `[batch, in_height, in_width, depth]`.
    filter: A `Tensor`. Must have the same type as `input`.
      3-D with shape `[filter_height, filter_width, depth]`.
    strides: A list of `ints` that has length `>= 4`.
      The stride of the sliding window for each dimension of the input
      tensor. Must be: `[1, stride_height, stride_width, 1]`.
    rates: A list of `ints` that has length `>= 4`.
      The input stride for atrous morphological dilation. Must be:
      `[1, rate_height, rate_width, 1]`.
    padding: A `string` from: `"SAME", "VALID"`.
      The type of padding algorithm to use.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "Dilation2D", name, input, filter, "strides", strides, "rates",
        rates, "padding", padding)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return dilation2d_eager_fallback(
          input, filter, strides=strides, rates=rates, padding=padding,
          name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if not isinstance(strides, (list, tuple)):
    raise TypeError(
        "Expected list for 'strides' argument to "
        "'dilation2d' Op, not %r." % strides)
  strides = [_execute.make_int(_i, "strides") for _i in strides]
  if not isinstance(rates, (list, tuple)):
    raise TypeError(
        "Expected list for 'rates' argument to "
        "'dilation2d' Op, not %r." % rates)
  rates = [_execute.make_int(_i, "rates") for _i in rates]
  padding = _execute.make_str(padding, "padding")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "Dilation2D", input=input, filter=filter, strides=strides,
                      rates=rates, padding=padding, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"), "strides",
              _op.get_attr("strides"), "rates", _op.get_attr("rates"),
              "padding", _op.get_attr("padding"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "Dilation2D", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

Dilation2D = tf_export("raw_ops.Dilation2D")(_ops.to_raw_op(dilation2d))


def dilation2d_eager_fallback(input: Annotated[Any, TV_Dilation2D_T], filter: Annotated[Any, TV_Dilation2D_T], strides, rates, padding: str, name, ctx) -> Annotated[Any, TV_Dilation2D_T]:
  if not isinstance(strides, (list, tuple)):
    raise TypeError(
        "Expected list for 'strides' argument to "
        "'dilation2d' Op, not %r." % strides)
  strides = [_execute.make_int(_i, "strides") for _i in strides]
  if not isinstance(rates, (list, tuple)):
    raise TypeError(
        "Expected list for 'rates' argument to "
        "'dilation2d' Op, not %r." % rates)
  rates = [_execute.make_int(_i, "rates") for _i in rates]
  padding = _execute.make_str(padding, "padding")
  _attr_T, _inputs_T = _execute.args_to_matching_eager([input, filter], ctx, [_dtypes.float32, _dtypes.float64, _dtypes.int32, _dtypes.uint8, _dtypes.int16, _dtypes.int8, _dtypes.int64, _dtypes.bfloat16, _dtypes.uint16, _dtypes.half, _dtypes.uint32, _dtypes.uint64, ])
  (input, filter) = _inputs_T
  _inputs_flat = [input, filter]
  _attrs = ("T", _attr_T, "strides", strides, "rates", rates, "padding",
  padding)
  _result = _execute.execute(b"Dilation2D", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "Dilation2D", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_Dilation2DBackpropFilter_T = TypeVar("TV_Dilation2DBackpropFilter_T", _atypes.BFloat16, _atypes.Float32, _atypes.Float64, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int64, _atypes.Int8, _atypes.UInt16, _atypes.UInt32, _atypes.UInt64, _atypes.UInt8)

def dilation2d_backprop_filter(input: Annotated[Any, TV_Dilation2DBackpropFilter_T], filter: Annotated[Any, TV_Dilation2DBackpropFilter_T], out_backprop: Annotated[Any, TV_Dilation2DBackpropFilter_T], strides, rates, padding: str, name=None) -> Annotated[Any, TV_Dilation2DBackpropFilter_T]:
  r"""Computes the gradient of morphological 2-D dilation with respect to the filter.

  Args:
    input: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`, `int64`, `bfloat16`, `uint16`, `half`, `uint32`, `uint64`.
      4-D with shape `[batch, in_height, in_width, depth]`.
    filter: A `Tensor`. Must have the same type as `input`.
      3-D with shape `[filter_height, filter_width, depth]`.
    out_backprop: A `Tensor`. Must have the same type as `input`.
      4-D with shape `[batch, out_height, out_width, depth]`.
    strides: A list of `ints` that has length `>= 4`.
      1-D of length 4. The stride of the sliding window for each dimension of
      the input tensor. Must be: `[1, stride_height, stride_width, 1]`.
    rates: A list of `ints` that has length `>= 4`.
      1-D of length 4. The input stride for atrous morphological dilation.
      Must be: `[1, rate_height, rate_width, 1]`.
    padding: A `string` from: `"SAME", "VALID"`.
      The type of padding algorithm to use.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "Dilation2DBackpropFilter", name, input, filter, out_backprop,
        "strides", strides, "rates", rates, "padding", padding)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return dilation2d_backprop_filter_eager_fallback(
          input, filter, out_backprop, strides=strides, rates=rates,
          padding=padding, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if not isinstance(strides, (list, tuple)):
    raise TypeError(
        "Expected list for 'strides' argument to "
        "'dilation2d_backprop_filter' Op, not %r." % strides)
  strides = [_execute.make_int(_i, "strides") for _i in strides]
  if not isinstance(rates, (list, tuple)):
    raise TypeError(
        "Expected list for 'rates' argument to "
        "'dilation2d_backprop_filter' Op, not %r." % rates)
  rates = [_execute.make_int(_i, "rates") for _i in rates]
  padding = _execute.make_str(padding, "padding")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "Dilation2DBackpropFilter", input=input, filter=filter,
                                    out_backprop=out_backprop,
                                    strides=strides, rates=rates,
                                    padding=padding, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"), "strides",
              _op.get_attr("strides"), "rates", _op.get_attr("rates"),
              "padding", _op.get_attr("padding"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "Dilation2DBackpropFilter", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

Dilation2DBackpropFilter = tf_export("raw_ops.Dilation2DBackpropFilter")(_ops.to_raw_op(dilation2d_backprop_filter))


def dilation2d_backprop_filter_eager_fallback(input: Annotated[Any, TV_Dilation2DBackpropFilter_T], filter: Annotated[Any, TV_Dilation2DBackpropFilter_T], out_backprop: Annotated[Any, TV_Dilation2DBackpropFilter_T], strides, rates, padding: str, name, ctx) -> Annotated[Any, TV_Dilation2DBackpropFilter_T]:
  if not isinstance(strides, (list, tuple)):
    raise TypeError(
        "Expected list for 'strides' argument to "
        "'dilation2d_backprop_filter' Op, not %r." % strides)
  strides = [_execute.make_int(_i, "strides") for _i in strides]
  if not isinstance(rates, (list, tuple)):
    raise TypeError(
        "Expected list for 'rates' argument to "
        "'dilation2d_backprop_filter' Op, not %r." % rates)
  rates = [_execute.make_int(_i, "rates") for _i in rates]
  padding = _execute.make_str(padding, "padding")
  _attr_T, _inputs_T = _execute.args_to_matching_eager([input, filter, out_backprop], ctx, [_dtypes.float32, _dtypes.float64, _dtypes.int32, _dtypes.uint8, _dtypes.int16, _dtypes.int8, _dtypes.int64, _dtypes.bfloat16, _dtypes.uint16, _dtypes.half, _dtypes.uint32, _dtypes.uint64, ])
  (input, filter, out_backprop) = _inputs_T
  _inputs_flat = [input, filter, out_backprop]
  _attrs = ("T", _attr_T, "strides", strides, "rates", rates, "padding",
  padding)
  _result = _execute.execute(b"Dilation2DBackpropFilter", 1,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "Dilation2DBackpropFilter", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_Dilation2DBackpropInput_T = TypeVar("TV_Dilation2DBackpropInput_T", _atypes.BFloat16, _atypes.Float32, _atypes.Float64, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int64, _atypes.Int8, _atypes.UInt16, _atypes.UInt32, _atypes.UInt64, _atypes.UInt8)

def dilation2d_backprop_input(input: Annotated[Any, TV_Dilation2DBackpropInput_T], filter: Annotated[Any, TV_Dilation2DBackpropInput_T], out_backprop: Annotated[Any, TV_Dilation2DBackpropInput_T], strides, rates, padding: str, name=None) -> Annotated[Any, TV_Dilation2DBackpropInput_T]:
  r"""Computes the gradient of morphological 2-D dilation with respect to the input.

  Args:
    input: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`, `int64`, `bfloat16`, `uint16`, `half`, `uint32`, `uint64`.
      4-D with shape `[batch, in_height, in_width, depth]`.
    filter: A `Tensor`. Must have the same type as `input`.
      3-D with shape `[filter_height, filter_width, depth]`.
    out_backprop: A `Tensor`. Must have the same type as `input`.
      4-D with shape `[batch, out_height, out_width, depth]`.
    strides: A list of `ints` that has length `>= 4`.
      1-D of length 4. The stride of the sliding window for each dimension of
      the input tensor. Must be: `[1, stride_height, stride_width, 1]`.
    rates: A list of `ints` that has length `>= 4`.
      1-D of length 4. The input stride for atrous morphological dilation.
      Must be: `[1, rate_height, rate_width, 1]`.
    padding: A `string` from: `"SAME", "VALID"`.
      The type of padding algorithm to use.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "Dilation2DBackpropInput", name, input, filter, out_backprop,
        "strides", strides, "rates", rates, "padding", padding)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return dilation2d_backprop_input_eager_fallback(
          input, filter, out_backprop, strides=strides, rates=rates,
          padding=padding, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if not isinstance(strides, (list, tuple)):
    raise TypeError(
        "Expected list for 'strides' argument to "
        "'dilation2d_backprop_input' Op, not %r." % strides)
  strides = [_execute.make_int(_i, "strides") for _i in strides]
  if not isinstance(rates, (list, tuple)):
    raise TypeError(
        "Expected list for 'rates' argument to "
        "'dilation2d_backprop_input' Op, not %r." % rates)
  rates = [_execute.make_int(_i, "rates") for _i in rates]
  padding = _execute.make_str(padding, "padding")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "Dilation2DBackpropInput", input=input, filter=filter,
                                   out_backprop=out_backprop, strides=strides,
                                   rates=rates, padding=padding, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"), "strides",
              _op.get_attr("strides"), "rates", _op.get_attr("rates"),
              "padding", _op.get_attr("padding"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "Dilation2DBackpropInput", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

Dilation2DBackpropInput = tf_export("raw_ops.Dilation2DBackpropInput")(_ops.to_raw_op(dilation2d_backprop_input))


def dilation2d_backprop_input_eager_fallback(input: Annotated[Any, TV_Dilation2DBackpropInput_T], filter: Annotated[Any, TV_Dilation2DBackpropInput_T], out_backprop: Annotated[Any, TV_Dilation2DBackpropInput_T], strides, rates, padding: str, name, ctx) -> Annotated[Any, TV_Dilation2DBackpropInput_T]:
  if not isinstance(strides, (list, tuple)):
    raise TypeError(
        "Expected list for 'strides' argument to "
        "'dilation2d_backprop_input' Op, not %r." % strides)
  strides = [_execute.make_int(_i, "strides") for _i in strides]
  if not isinstance(rates, (list, tuple)):
    raise TypeError(
        "Expected list for 'rates' argument to "
        "'dilation2d_backprop_input' Op, not %r." % rates)
  rates = [_execute.make_int(_i, "rates") for _i in rates]
  padding = _execute.make_str(padding, "padding")
  _attr_T, _inputs_T = _execute.args_to_matching_eager([input, filter, out_backprop], ctx, [_dtypes.float32, _dtypes.float64, _dtypes.int32, _dtypes.uint8, _dtypes.int16, _dtypes.int8, _dtypes.int64, _dtypes.bfloat16, _dtypes.uint16, _dtypes.half, _dtypes.uint32, _dtypes.uint64, ])
  (input, filter, out_backprop) = _inputs_T
  _inputs_flat = [input, filter, out_backprop]
  _attrs = ("T", _attr_T, "strides", strides, "rates", rates, "padding",
  padding)
  _result = _execute.execute(b"Dilation2DBackpropInput", 1,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "Dilation2DBackpropInput", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_Elu_T = TypeVar("TV_Elu_T", _atypes.BFloat16, _atypes.Float32, _atypes.Float64, _atypes.Half)

@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('nn.elu')
def elu(features: Annotated[Any, TV_Elu_T], name=None) -> Annotated[Any, TV_Elu_T]:
  r"""Computes the exponential linear function.

  The ELU function is defined as:

   * $ e ^ x - 1 $ if $ x < 0 $
   * $ x $ if $ x >= 0 $

  Examples:

  >>> tf.nn.elu(1.0)
  <tf.Tensor: shape=(), dtype=float32, numpy=1.0>
  >>> tf.nn.elu(0.0)
  <tf.Tensor: shape=(), dtype=float32, numpy=0.0>
  >>> tf.nn.elu(-1000.0)
  <tf.Tensor: shape=(), dtype=float32, numpy=-1.0>

  See [Fast and Accurate Deep Network Learning by Exponential Linear Units (ELUs)
  ](http://arxiv.org/abs/1511.07289)

  Args:
    features: A `Tensor`. Must be one of the following types: `half`, `bfloat16`, `float32`, `float64`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `features`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "Elu", name, features)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_elu(
          (features, name,), None)
      if _result is not NotImplemented:
        return _result
      return elu_eager_fallback(
          features, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            elu, (), dict(features=features, name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_elu(
        (features, name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "Elu", features=features, name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          elu, (), dict(features=features, name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "Elu", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

Elu = tf_export("raw_ops.Elu")(_ops.to_raw_op(elu))
_dispatcher_for_elu = elu._tf_type_based_dispatcher.Dispatch


def elu_eager_fallback(features: Annotated[Any, TV_Elu_T], name, ctx) -> Annotated[Any, TV_Elu_T]:
  _attr_T, (features,) = _execute.args_to_matching_eager([features], ctx, [_dtypes.half, _dtypes.bfloat16, _dtypes.float32, _dtypes.float64, ])
  _inputs_flat = [features]
  _attrs = ("T", _attr_T)
  _result = _execute.execute(b"Elu", 1, inputs=_inputs_flat, attrs=_attrs,
                             ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "Elu", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_EluGrad_T = TypeVar("TV_EluGrad_T", _atypes.BFloat16, _atypes.Float32, _atypes.Float64, _atypes.Half)

def elu_grad(gradients: Annotated[Any, TV_EluGrad_T], outputs: Annotated[Any, TV_EluGrad_T], name=None) -> Annotated[Any, TV_EluGrad_T]:
  r"""Computes gradients for the exponential linear (Elu) operation.

  Args:
    gradients: A `Tensor`. Must be one of the following types: `half`, `bfloat16`, `float32`, `float64`.
      The backpropagated gradients to the corresponding Elu operation.
    outputs: A `Tensor`. Must have the same type as `gradients`.
      The outputs of the corresponding Elu operation.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `gradients`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "EluGrad", name, gradients, outputs)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return elu_grad_eager_fallback(
          gradients, outputs, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "EluGrad", gradients=gradients, outputs=outputs, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "EluGrad", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

EluGrad = tf_export("raw_ops.EluGrad")(_ops.to_raw_op(elu_grad))


def elu_grad_eager_fallback(gradients: Annotated[Any, TV_EluGrad_T], outputs: Annotated[Any, TV_EluGrad_T], name, ctx) -> Annotated[Any, TV_EluGrad_T]:
  _attr_T, _inputs_T = _execute.args_to_matching_eager([gradients, outputs], ctx, [_dtypes.half, _dtypes.bfloat16, _dtypes.float32, _dtypes.float64, ])
  (gradients, outputs) = _inputs_T
  _inputs_flat = [gradients, outputs]
  _attrs = ("T", _attr_T)
  _result = _execute.execute(b"EluGrad", 1, inputs=_inputs_flat, attrs=_attrs,
                             ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "EluGrad", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

_FractionalAvgPoolOutput = collections.namedtuple(
    "FractionalAvgPool",
    ["output", "row_pooling_sequence", "col_pooling_sequence"])


TV_FractionalAvgPool_T = TypeVar("TV_FractionalAvgPool_T", _atypes.Float32, _atypes.Float64, _atypes.Int32, _atypes.Int64)

def fractional_avg_pool(value: Annotated[Any, TV_FractionalAvgPool_T], pooling_ratio, pseudo_random:bool=False, overlapping:bool=False, deterministic:bool=False, seed:int=0, seed2:int=0, name=None):
  r"""Performs fractional average pooling on the input.

  Fractional average pooling is similar to Fractional max pooling in the pooling
  region generation step. The only difference is that after pooling regions are
  generated, a mean operation is performed instead of a max operation in each
  pooling region.

  Args:
    value: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `int64`.
      4-D with shape `[batch, height, width, channels]`.
    pooling_ratio: A list of `floats` that has length `>= 4`.
      Pooling ratio for each dimension of `value`, currently only
      supports row and col dimension and should be >= 1.0. For example, a valid
      pooling ratio looks like [1.0, 1.44, 1.73, 1.0]. The first and last elements
      must be 1.0 because we don't allow pooling on batch and channels
      dimensions. 1.44 and 1.73 are pooling ratio on height and width dimensions
      respectively.
    pseudo_random: An optional `bool`. Defaults to `False`.
      When set to True, generates the pooling sequence in a
      pseudorandom fashion, otherwise, in a random fashion. Check paper [Benjamin
      Graham, Fractional Max-Pooling](http://arxiv.org/abs/1412.6071) for
      difference between pseudorandom and random.
    overlapping: An optional `bool`. Defaults to `False`.
      When set to True, it means when pooling, the values at the boundary
      of adjacent pooling cells are used by both cells. For example:

      `index  0  1  2  3  4`

      `value  20 5  16 3  7`

      If the pooling sequence is [0, 2, 4], then 16, at index 2 will be used twice.
      The result would be [41/3, 26/3] for fractional avg pooling.
    deterministic: An optional `bool`. Defaults to `False`.
      When set to True, a fixed pooling region will be used when
      iterating over a FractionalAvgPool node in the computation graph. Mainly used
      in unit test to make FractionalAvgPool deterministic.
    seed: An optional `int`. Defaults to `0`.
      If either seed or seed2 are set to be non-zero, the random number
      generator is seeded by the given seed.  Otherwise, it is seeded by a
      random seed.
    seed2: An optional `int`. Defaults to `0`.
      An second seed to avoid seed collision.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (output, row_pooling_sequence, col_pooling_sequence).

    output: A `Tensor`. Has the same type as `value`.
    row_pooling_sequence: A `Tensor` of type `int64`.
    col_pooling_sequence: A `Tensor` of type `int64`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "FractionalAvgPool", name, value, "pooling_ratio",
        pooling_ratio, "pseudo_random", pseudo_random, "overlapping",
        overlapping, "deterministic", deterministic, "seed", seed, "seed2",
        seed2)
      _result = _FractionalAvgPoolOutput._make(_result)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return fractional_avg_pool_eager_fallback(
          value, pooling_ratio=pooling_ratio, pseudo_random=pseudo_random,
          overlapping=overlapping, deterministic=deterministic, seed=seed,
          seed2=seed2, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if not isinstance(pooling_ratio, (list, tuple)):
    raise TypeError(
        "Expected list for 'pooling_ratio' argument to "
        "'fractional_avg_pool' Op, not %r." % pooling_ratio)
  pooling_ratio = [_execute.make_float(_f, "pooling_ratio") for _f in pooling_ratio]
  if pseudo_random is None:
    pseudo_random = False
  pseudo_random = _execute.make_bool(pseudo_random, "pseudo_random")
  if overlapping is None:
    overlapping = False
  overlapping = _execute.make_bool(overlapping, "overlapping")
  if deterministic is None:
    deterministic = False
  deterministic = _execute.make_bool(deterministic, "deterministic")
  if seed is None:
    seed = 0
  seed = _execute.make_int(seed, "seed")
  if seed2 is None:
    seed2 = 0
  seed2 = _execute.make_int(seed2, "seed2")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "FractionalAvgPool", value=value, pooling_ratio=pooling_ratio,
                             pseudo_random=pseudo_random,
                             overlapping=overlapping,
                             deterministic=deterministic, seed=seed,
                             seed2=seed2, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("pooling_ratio", _op.get_attr("pooling_ratio"), "pseudo_random",
              _op._get_attr_bool("pseudo_random"), "overlapping",
              _op._get_attr_bool("overlapping"), "deterministic",
              _op._get_attr_bool("deterministic"), "seed",
              _op._get_attr_int("seed"), "seed2", _op._get_attr_int("seed2"),
              "T", _op._get_attr_type("T"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "FractionalAvgPool", _inputs_flat, _attrs, _result)
  _result = _FractionalAvgPoolOutput._make(_result)
  return _result

FractionalAvgPool = tf_export("raw_ops.FractionalAvgPool")(_ops.to_raw_op(fractional_avg_pool))


def fractional_avg_pool_eager_fallback(value: Annotated[Any, TV_FractionalAvgPool_T], pooling_ratio, pseudo_random: bool, overlapping: bool, deterministic: bool, seed: int, seed2: int, name, ctx):
  if not isinstance(pooling_ratio, (list, tuple)):
    raise TypeError(
        "Expected list for 'pooling_ratio' argument to "
        "'fractional_avg_pool' Op, not %r." % pooling_ratio)
  pooling_ratio = [_execute.make_float(_f, "pooling_ratio") for _f in pooling_ratio]
  if pseudo_random is None:
    pseudo_random = False
  pseudo_random = _execute.make_bool(pseudo_random, "pseudo_random")
  if overlapping is None:
    overlapping = False
  overlapping = _execute.make_bool(overlapping, "overlapping")
  if deterministic is None:
    deterministic = False
  deterministic = _execute.make_bool(deterministic, "deterministic")
  if seed is None:
    seed = 0
  seed = _execute.make_int(seed, "seed")
  if seed2 is None:
    seed2 = 0
  seed2 = _execute.make_int(seed2, "seed2")
  _attr_T, (value,) = _execute.args_to_matching_eager([value], ctx, [_dtypes.float32, _dtypes.float64, _dtypes.int32, _dtypes.int64, ])
  _inputs_flat = [value]
  _attrs = ("pooling_ratio", pooling_ratio, "pseudo_random", pseudo_random,
  "overlapping", overlapping, "deterministic", deterministic, "seed", seed,
  "seed2", seed2, "T", _attr_T)
  _result = _execute.execute(b"FractionalAvgPool", 3, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "FractionalAvgPool", _inputs_flat, _attrs, _result)
  _result = _FractionalAvgPoolOutput._make(_result)
  return _result


TV_FractionalAvgPoolGrad_T = TypeVar("TV_FractionalAvgPoolGrad_T", _atypes.Float32, _atypes.Float64, _atypes.Int32, _atypes.Int64)

def fractional_avg_pool_grad(orig_input_tensor_shape: Annotated[Any, _atypes.Int64], out_backprop: Annotated[Any, TV_FractionalAvgPoolGrad_T], row_pooling_sequence: Annotated[Any, _atypes.Int64], col_pooling_sequence: Annotated[Any, _atypes.Int64], overlapping:bool=False, name=None) -> Annotated[Any, TV_FractionalAvgPoolGrad_T]:
  r"""Computes gradient of the FractionalAvgPool function.

  Unlike FractionalMaxPoolGrad, we don't need to find arg_max for
  FractionalAvgPoolGrad, we just need to evenly back-propagate each element of
  out_backprop to those indices that form the same pooling cell. Therefore, we
  just need to know the shape of original input tensor, instead of the whole
  tensor.

  Args:
    orig_input_tensor_shape: A `Tensor` of type `int64`.
      Original input tensor shape for `fractional_avg_pool`
    out_backprop: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `int64`.
      4-D with shape `[batch, height, width, channels]`.  Gradients
      w.r.t. the output of `fractional_avg_pool`.
    row_pooling_sequence: A `Tensor` of type `int64`.
      row pooling sequence, form pooling region with
      col_pooling_sequence.
    col_pooling_sequence: A `Tensor` of type `int64`.
      column pooling sequence, form pooling region with
      row_pooling sequence.
    overlapping: An optional `bool`. Defaults to `False`.
      When set to True, it means when pooling, the values at the boundary
      of adjacent pooling cells are used by both cells. For example:

      `index  0  1  2  3  4`

      `value  20 5  16 3  7`

      If the pooling sequence is [0, 2, 4], then 16, at index 2 will be used twice.
      The result would be [41/3, 26/3] for fractional avg pooling.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `out_backprop`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "FractionalAvgPoolGrad", name, orig_input_tensor_shape,
        out_backprop, row_pooling_sequence, col_pooling_sequence,
        "overlapping", overlapping)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return fractional_avg_pool_grad_eager_fallback(
          orig_input_tensor_shape, out_backprop, row_pooling_sequence,
          col_pooling_sequence, overlapping=overlapping, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if overlapping is None:
    overlapping = False
  overlapping = _execute.make_bool(overlapping, "overlapping")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "FractionalAvgPoolGrad", orig_input_tensor_shape=orig_input_tensor_shape,
                                 out_backprop=out_backprop,
                                 row_pooling_sequence=row_pooling_sequence,
                                 col_pooling_sequence=col_pooling_sequence,
                                 overlapping=overlapping, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("overlapping", _op._get_attr_bool("overlapping"), "T",
              _op._get_attr_type("T"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "FractionalAvgPoolGrad", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

FractionalAvgPoolGrad = tf_export("raw_ops.FractionalAvgPoolGrad")(_ops.to_raw_op(fractional_avg_pool_grad))


def fractional_avg_pool_grad_eager_fallback(orig_input_tensor_shape: Annotated[Any, _atypes.Int64], out_backprop: Annotated[Any, TV_FractionalAvgPoolGrad_T], row_pooling_sequence: Annotated[Any, _atypes.Int64], col_pooling_sequence: Annotated[Any, _atypes.Int64], overlapping: bool, name, ctx) -> Annotated[Any, TV_FractionalAvgPoolGrad_T]:
  if overlapping is None:
    overlapping = False
  overlapping = _execute.make_bool(overlapping, "overlapping")
  _attr_T, (out_backprop,) = _execute.args_to_matching_eager([out_backprop], ctx, [_dtypes.float32, _dtypes.float64, _dtypes.int32, _dtypes.int64, ])
  orig_input_tensor_shape = _ops.convert_to_tensor(orig_input_tensor_shape, _dtypes.int64)
  row_pooling_sequence = _ops.convert_to_tensor(row_pooling_sequence, _dtypes.int64)
  col_pooling_sequence = _ops.convert_to_tensor(col_pooling_sequence, _dtypes.int64)
  _inputs_flat = [orig_input_tensor_shape, out_backprop, row_pooling_sequence, col_pooling_sequence]
  _attrs = ("overlapping", overlapping, "T", _attr_T)
  _result = _execute.execute(b"FractionalAvgPoolGrad", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "FractionalAvgPoolGrad", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

_FractionalMaxPoolOutput = collections.namedtuple(
    "FractionalMaxPool",
    ["output", "row_pooling_sequence", "col_pooling_sequence"])


TV_FractionalMaxPool_T = TypeVar("TV_FractionalMaxPool_T", _atypes.Float32, _atypes.Float64, _atypes.Int32, _atypes.Int64)

def fractional_max_pool(value: Annotated[Any, TV_FractionalMaxPool_T], pooling_ratio, pseudo_random:bool=False, overlapping:bool=False, deterministic:bool=False, seed:int=0, seed2:int=0, name=None):
  r"""Performs fractional max pooling on the input.

  Fractional max pooling is slightly different than regular max pooling.  In
  regular max pooling, you downsize an input set by taking the maximum value of
  smaller N x N subsections of the set (often 2x2), and try to reduce the set by
  a factor of N, where N is an integer.  Fractional max pooling, as you might
  expect from the word "fractional", means that the overall reduction ratio N
  does not have to be an integer.

  The sizes of the pooling regions are generated randomly but are fairly uniform.
  For example, let's look at the height dimension, and the constraints on the
  list of rows that will be pool boundaries.

  First we define the following:

  1.  input_row_length : the number of rows from the input set
  2.  output_row_length : which will be smaller than the input
  3.  alpha = input_row_length / output_row_length : our reduction ratio
  4.  K = floor(alpha)
  5.  row_pooling_sequence : this is the result list of pool boundary rows

  Then, row_pooling_sequence should satisfy:

  1.  a[0] = 0 : the first value of the sequence is 0
  2.  a[end] = input_row_length : the last value of the sequence is the size
  3.  K <= (a[i+1] - a[i]) <= K+1 : all intervals are K or K+1 size
  4.  length(row_pooling_sequence) = output_row_length+1

  For more details on fractional max pooling, see this paper:
  [Benjamin Graham, Fractional Max-Pooling](http://arxiv.org/abs/1412.6071)

  Args:
    value: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `int64`.
      4-D with shape `[batch, height, width, channels]`.
    pooling_ratio: A list of `floats` that has length `>= 4`.
      Pooling ratio for each dimension of `value`, currently only
      supports row and col dimension and should be >= 1.0. For example, a valid
      pooling ratio looks like [1.0, 1.44, 1.73, 1.0]. The first and last elements
      must be 1.0 because we don't allow pooling on batch and channels
      dimensions. 1.44 and 1.73 are pooling ratio on height and width dimensions
      respectively.
    pseudo_random: An optional `bool`. Defaults to `False`.
      When set to True, generates the pooling sequence in a
      pseudorandom fashion, otherwise, in a random fashion. Check paper [Benjamin
      Graham, Fractional Max-Pooling](http://arxiv.org/abs/1412.6071) for
      difference between pseudorandom and random.
    overlapping: An optional `bool`. Defaults to `False`.
      When set to True, it means when pooling, the values at the boundary
      of adjacent pooling cells are used by both cells. For example:

      `index  0  1  2  3  4`

      `value  20 5  16 3  7`

      If the pooling sequence is [0, 2, 4], then 16, at index 2 will be used twice.
      The result would be [20, 16] for fractional max pooling.
    deterministic: An optional `bool`. Defaults to `False`.
      When set to True, a fixed pooling region will be used when
      iterating over a FractionalMaxPool node in the computation graph. Mainly used
      in unit test to make FractionalMaxPool deterministic.
    seed: An optional `int`. Defaults to `0`.
      If either seed or seed2 are set to be non-zero, the random number
      generator is seeded by the given seed.  Otherwise, it is seeded by a
      random seed.
    seed2: An optional `int`. Defaults to `0`.
      An second seed to avoid seed collision.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (output, row_pooling_sequence, col_pooling_sequence).

    output: A `Tensor`. Has the same type as `value`.
    row_pooling_sequence: A `Tensor` of type `int64`.
    col_pooling_sequence: A `Tensor` of type `int64`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "FractionalMaxPool", name, value, "pooling_ratio",
        pooling_ratio, "pseudo_random", pseudo_random, "overlapping",
        overlapping, "deterministic", deterministic, "seed", seed, "seed2",
        seed2)
      _result = _FractionalMaxPoolOutput._make(_result)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return fractional_max_pool_eager_fallback(
          value, pooling_ratio=pooling_ratio, pseudo_random=pseudo_random,
          overlapping=overlapping, deterministic=deterministic, seed=seed,
          seed2=seed2, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if not isinstance(pooling_ratio, (list, tuple)):
    raise TypeError(
        "Expected list for 'pooling_ratio' argument to "
        "'fractional_max_pool' Op, not %r." % pooling_ratio)
  pooling_ratio = [_execute.make_float(_f, "pooling_ratio") for _f in pooling_ratio]
  if pseudo_random is None:
    pseudo_random = False
  pseudo_random = _execute.make_bool(pseudo_random, "pseudo_random")
  if overlapping is None:
    overlapping = False
  overlapping = _execute.make_bool(overlapping, "overlapping")
  if deterministic is None:
    deterministic = False
  deterministic = _execute.make_bool(deterministic, "deterministic")
  if seed is None:
    seed = 0
  seed = _execute.make_int(seed, "seed")
  if seed2 is None:
    seed2 = 0
  seed2 = _execute.make_int(seed2, "seed2")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "FractionalMaxPool", value=value, pooling_ratio=pooling_ratio,
                             pseudo_random=pseudo_random,
                             overlapping=overlapping,
                             deterministic=deterministic, seed=seed,
                             seed2=seed2, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("pooling_ratio", _op.get_attr("pooling_ratio"), "pseudo_random",
              _op._get_attr_bool("pseudo_random"), "overlapping",
              _op._get_attr_bool("overlapping"), "deterministic",
              _op._get_attr_bool("deterministic"), "seed",
              _op._get_attr_int("seed"), "seed2", _op._get_attr_int("seed2"),
              "T", _op._get_attr_type("T"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "FractionalMaxPool", _inputs_flat, _attrs, _result)
  _result = _FractionalMaxPoolOutput._make(_result)
  return _result

FractionalMaxPool = tf_export("raw_ops.FractionalMaxPool")(_ops.to_raw_op(fractional_max_pool))


def fractional_max_pool_eager_fallback(value: Annotated[Any, TV_FractionalMaxPool_T], pooling_ratio, pseudo_random: bool, overlapping: bool, deterministic: bool, seed: int, seed2: int, name, ctx):
  if not isinstance(pooling_ratio, (list, tuple)):
    raise TypeError(
        "Expected list for 'pooling_ratio' argument to "
        "'fractional_max_pool' Op, not %r." % pooling_ratio)
  pooling_ratio = [_execute.make_float(_f, "pooling_ratio") for _f in pooling_ratio]
  if pseudo_random is None:
    pseudo_random = False
  pseudo_random = _execute.make_bool(pseudo_random, "pseudo_random")
  if overlapping is None:
    overlapping = False
  overlapping = _execute.make_bool(overlapping, "overlapping")
  if deterministic is None:
    deterministic = False
  deterministic = _execute.make_bool(deterministic, "deterministic")
  if seed is None:
    seed = 0
  seed = _execute.make_int(seed, "seed")
  if seed2 is None:
    seed2 = 0
  seed2 = _execute.make_int(seed2, "seed2")
  _attr_T, (value,) = _execute.args_to_matching_eager([value], ctx, [_dtypes.float32, _dtypes.float64, _dtypes.int32, _dtypes.int64, ])
  _inputs_flat = [value]
  _attrs = ("pooling_ratio", pooling_ratio, "pseudo_random", pseudo_random,
  "overlapping", overlapping, "deterministic", deterministic, "seed", seed,
  "seed2", seed2, "T", _attr_T)
  _result = _execute.execute(b"FractionalMaxPool", 3, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "FractionalMaxPool", _inputs_flat, _attrs, _result)
  _result = _FractionalMaxPoolOutput._make(_result)
  return _result


TV_FractionalMaxPoolGrad_T = TypeVar("TV_FractionalMaxPoolGrad_T", _atypes.Float32, _atypes.Float64, _atypes.Int32, _atypes.Int64)

def fractional_max_pool_grad(orig_input: Annotated[Any, TV_FractionalMaxPoolGrad_T], orig_output: Annotated[Any, TV_FractionalMaxPoolGrad_T], out_backprop: Annotated[Any, TV_FractionalMaxPoolGrad_T], row_pooling_sequence: Annotated[Any, _atypes.Int64], col_pooling_sequence: Annotated[Any, _atypes.Int64], overlapping:bool=False, name=None) -> Annotated[Any, TV_FractionalMaxPoolGrad_T]:
  r"""Computes gradient of the FractionalMaxPool function.

  Args:
    orig_input: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `int64`.
      Original input for `fractional_max_pool`
    orig_output: A `Tensor`. Must have the same type as `orig_input`.
      Original output for `fractional_max_pool`
    out_backprop: A `Tensor`. Must have the same type as `orig_input`.
      4-D with shape `[batch, height, width, channels]`.  Gradients
      w.r.t. the output of `fractional_max_pool`.
    row_pooling_sequence: A `Tensor` of type `int64`.
      row pooling sequence, form pooling region with
      col_pooling_sequence.
    col_pooling_sequence: A `Tensor` of type `int64`.
      column pooling sequence, form pooling region with
      row_pooling sequence.
    overlapping: An optional `bool`. Defaults to `False`.
      When set to True, it means when pooling, the values at the boundary
      of adjacent pooling cells are used by both cells. For example:

      `index  0  1  2  3  4`

      `value  20 5  16 3  7`

      If the pooling sequence is [0, 2, 4], then 16, at index 2 will be used twice.
      The result would be [20, 16] for fractional max pooling.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `orig_input`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "FractionalMaxPoolGrad", name, orig_input, orig_output,
        out_backprop, row_pooling_sequence, col_pooling_sequence,
        "overlapping", overlapping)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return fractional_max_pool_grad_eager_fallback(
          orig_input, orig_output, out_backprop, row_pooling_sequence,
          col_pooling_sequence, overlapping=overlapping, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if overlapping is None:
    overlapping = False
  overlapping = _execute.make_bool(overlapping, "overlapping")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "FractionalMaxPoolGrad", orig_input=orig_input,
                                 orig_output=orig_output,
                                 out_backprop=out_backprop,
                                 row_pooling_sequence=row_pooling_sequence,
                                 col_pooling_sequence=col_pooling_sequence,
                                 overlapping=overlapping, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("overlapping", _op._get_attr_bool("overlapping"), "T",
              _op._get_attr_type("T"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "FractionalMaxPoolGrad", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

FractionalMaxPoolGrad = tf_export("raw_ops.FractionalMaxPoolGrad")(_ops.to_raw_op(fractional_max_pool_grad))


def fractional_max_pool_grad_eager_fallback(orig_input: Annotated[Any, TV_FractionalMaxPoolGrad_T], orig_output: Annotated[Any, TV_FractionalMaxPoolGrad_T], out_backprop: Annotated[Any, TV_FractionalMaxPoolGrad_T], row_pooling_sequence: Annotated[Any, _atypes.Int64], col_pooling_sequence: Annotated[Any, _atypes.Int64], overlapping: bool, name, ctx) -> Annotated[Any, TV_FractionalMaxPoolGrad_T]:
  if overlapping is None:
    overlapping = False
  overlapping = _execute.make_bool(overlapping, "overlapping")
  _attr_T, _inputs_T = _execute.args_to_matching_eager([orig_input, orig_output, out_backprop], ctx, [_dtypes.float32, _dtypes.float64, _dtypes.int32, _dtypes.int64, ])
  (orig_input, orig_output, out_backprop) = _inputs_T
  row_pooling_sequence = _ops.convert_to_tensor(row_pooling_sequence, _dtypes.int64)
  col_pooling_sequence = _ops.convert_to_tensor(col_pooling_sequence, _dtypes.int64)
  _inputs_flat = [orig_input, orig_output, out_backprop, row_pooling_sequence, col_pooling_sequence]
  _attrs = ("overlapping", overlapping, "T", _attr_T)
  _result = _execute.execute(b"FractionalMaxPoolGrad", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "FractionalMaxPoolGrad", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

_FusedBatchNormOutput = collections.namedtuple(
    "FusedBatchNorm",
    ["y", "batch_mean", "batch_variance", "reserve_space_1", "reserve_space_2"])


TV_FusedBatchNorm_T = TypeVar("TV_FusedBatchNorm_T", bound=_atypes.Float32)

def _fused_batch_norm(x: Annotated[Any, TV_FusedBatchNorm_T], scale: Annotated[Any, TV_FusedBatchNorm_T], offset: Annotated[Any, TV_FusedBatchNorm_T], mean: Annotated[Any, TV_FusedBatchNorm_T], variance: Annotated[Any, TV_FusedBatchNorm_T], epsilon:float=0.0001, exponential_avg_factor:float=1, data_format:str="NHWC", is_training:bool=True, name=None):
  r"""Batch normalization.

  Note that the size of 4D Tensors are defined by either "NHWC" or "NCHW".
  The size of 1D Tensors matches the dimension C of the 4D Tensors.

  Args:
    x: A `Tensor`. Must be one of the following types: `float32`.
      A 4D Tensor for input data.
    scale: A `Tensor`. Must have the same type as `x`.
      A 1D Tensor for scaling factor, to scale the normalized x.
    offset: A `Tensor`. Must have the same type as `x`.
      A 1D Tensor for offset, to shift to the normalized x.
    mean: A `Tensor`. Must have the same type as `x`.
      A 1D Tensor for population mean. Used for inference only;
      must be empty for training.
    variance: A `Tensor`. Must have the same type as `x`.
      A 1D Tensor for population variance. Used for inference only;
      must be empty for training.
    epsilon: An optional `float`. Defaults to `0.0001`.
      A small float number added to the variance of x.
    exponential_avg_factor: An optional `float`. Defaults to `1`.
    data_format: An optional `string` from: `"NHWC", "NCHW"`. Defaults to `"NHWC"`.
      The data format for x and y. Either "NHWC" (default) or "NCHW".
    is_training: An optional `bool`. Defaults to `True`.
      A bool value to indicate the operation is for training (default)
      or inference.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (y, batch_mean, batch_variance, reserve_space_1, reserve_space_2).

    y: A `Tensor`. Has the same type as `x`.
    batch_mean: A `Tensor`. Has the same type as `x`.
    batch_variance: A `Tensor`. Has the same type as `x`.
    reserve_space_1: A `Tensor`. Has the same type as `x`.
    reserve_space_2: A `Tensor`. Has the same type as `x`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "FusedBatchNorm", name, x, scale, offset, mean, variance,
        "epsilon", epsilon, "exponential_avg_factor", exponential_avg_factor,
        "data_format", data_format, "is_training", is_training)
      _result = _FusedBatchNormOutput._make(_result)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return _fused_batch_norm_eager_fallback(
          x, scale, offset, mean, variance, epsilon=epsilon,
          exponential_avg_factor=exponential_avg_factor,
          data_format=data_format, is_training=is_training, name=name,
          ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if epsilon is None:
    epsilon = 0.0001
  epsilon = _execute.make_float(epsilon, "epsilon")
  if exponential_avg_factor is None:
    exponential_avg_factor = 1
  exponential_avg_factor = _execute.make_float(exponential_avg_factor, "exponential_avg_factor")
  if data_format is None:
    data_format = "NHWC"
  data_format = _execute.make_str(data_format, "data_format")
  if is_training is None:
    is_training = True
  is_training = _execute.make_bool(is_training, "is_training")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "FusedBatchNorm", x=x, scale=scale, offset=offset, mean=mean,
                          variance=variance, epsilon=epsilon,
                          exponential_avg_factor=exponential_avg_factor,
                          data_format=data_format, is_training=is_training,
                          name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"), "epsilon",
              _op.get_attr("epsilon"), "exponential_avg_factor",
              _op.get_attr("exponential_avg_factor"), "data_format",
              _op.get_attr("data_format"), "is_training",
              _op._get_attr_bool("is_training"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "FusedBatchNorm", _inputs_flat, _attrs, _result)
  _result = _FusedBatchNormOutput._make(_result)
  return _result

FusedBatchNorm = tf_export("raw_ops.FusedBatchNorm")(_ops.to_raw_op(_fused_batch_norm))


def _fused_batch_norm_eager_fallback(x: Annotated[Any, TV_FusedBatchNorm_T], scale: Annotated[Any, TV_FusedBatchNorm_T], offset: Annotated[Any, TV_FusedBatchNorm_T], mean: Annotated[Any, TV_FusedBatchNorm_T], variance: Annotated[Any, TV_FusedBatchNorm_T], epsilon: float, exponential_avg_factor: float, data_format: str, is_training: bool, name, ctx):
  if epsilon is None:
    epsilon = 0.0001
  epsilon = _execute.make_float(epsilon, "epsilon")
  if exponential_avg_factor is None:
    exponential_avg_factor = 1
  exponential_avg_factor = _execute.make_float(exponential_avg_factor, "exponential_avg_factor")
  if data_format is None:
    data_format = "NHWC"
  data_format = _execute.make_str(data_format, "data_format")
  if is_training is None:
    is_training = True
  is_training = _execute.make_bool(is_training, "is_training")
  _attr_T, _inputs_T = _execute.args_to_matching_eager([x, scale, offset, mean, variance], ctx, [_dtypes.float32, ])
  (x, scale, offset, mean, variance) = _inputs_T
  _inputs_flat = [x, scale, offset, mean, variance]
  _attrs = ("T", _attr_T, "epsilon", epsilon, "exponential_avg_factor",
  exponential_avg_factor, "data_format", data_format, "is_training",
  is_training)
  _result = _execute.execute(b"FusedBatchNorm", 5, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "FusedBatchNorm", _inputs_flat, _attrs, _result)
  _result = _FusedBatchNormOutput._make(_result)
  return _result

_FusedBatchNormGradOutput = collections.namedtuple(
    "FusedBatchNormGrad",
    ["x_backprop", "scale_backprop", "offset_backprop", "reserve_space_3", "reserve_space_4"])


TV_FusedBatchNormGrad_T = TypeVar("TV_FusedBatchNormGrad_T", bound=_atypes.Float32)

def fused_batch_norm_grad(y_backprop: Annotated[Any, TV_FusedBatchNormGrad_T], x: Annotated[Any, TV_FusedBatchNormGrad_T], scale: Annotated[Any, TV_FusedBatchNormGrad_T], reserve_space_1: Annotated[Any, TV_FusedBatchNormGrad_T], reserve_space_2: Annotated[Any, TV_FusedBatchNormGrad_T], epsilon:float=0.0001, data_format:str="NHWC", is_training:bool=True, name=None):
  r"""Gradient for batch normalization.

  Note that the size of 4D Tensors are defined by either "NHWC" or "NCHW".
  The size of 1D Tensors matches the dimension C of the 4D Tensors.

  Args:
    y_backprop: A `Tensor`. Must be one of the following types: `float32`.
      A 4D Tensor for the gradient with respect to y.
    x: A `Tensor`. Must have the same type as `y_backprop`.
      A 4D Tensor for input data.
    scale: A `Tensor`. Must have the same type as `y_backprop`.
      A 1D Tensor for scaling factor, to scale the normalized x.
    reserve_space_1: A `Tensor`. Must have the same type as `y_backprop`.
      When is_training is True, a 1D Tensor for the computed batch
      mean to be reused in gradient computation. When is_training is
      False, a 1D Tensor for the population mean to be reused in both
      1st and 2nd order gradient computation.
    reserve_space_2: A `Tensor`. Must have the same type as `y_backprop`.
      When is_training is True, a 1D Tensor for the computed batch
      variance (inverted variance in the cuDNN case) to be reused in
      gradient computation. When is_training is False, a 1D Tensor
      for the population variance to be reused in both 1st and 2nd
      order gradient computation.
    epsilon: An optional `float`. Defaults to `0.0001`.
      A small float number added to the variance of x.
    data_format: An optional `string` from: `"NHWC", "NCHW"`. Defaults to `"NHWC"`.
      The data format for y_backprop, x, x_backprop.
      Either "NHWC" (default) or "NCHW".
    is_training: An optional `bool`. Defaults to `True`.
      A bool value to indicate the operation is for training (default)
      or inference.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (x_backprop, scale_backprop, offset_backprop, reserve_space_3, reserve_space_4).

    x_backprop: A `Tensor`. Has the same type as `y_backprop`.
    scale_backprop: A `Tensor`. Has the same type as `y_backprop`.
    offset_backprop: A `Tensor`. Has the same type as `y_backprop`.
    reserve_space_3: A `Tensor`. Has the same type as `y_backprop`.
    reserve_space_4: A `Tensor`. Has the same type as `y_backprop`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "FusedBatchNormGrad", name, y_backprop, x, scale,
        reserve_space_1, reserve_space_2, "epsilon", epsilon, "data_format",
        data_format, "is_training", is_training)
      _result = _FusedBatchNormGradOutput._make(_result)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return fused_batch_norm_grad_eager_fallback(
          y_backprop, x, scale, reserve_space_1, reserve_space_2,
          epsilon=epsilon, data_format=data_format, is_training=is_training,
          name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if epsilon is None:
    epsilon = 0.0001
  epsilon = _execute.make_float(epsilon, "epsilon")
  if data_format is None:
    data_format = "NHWC"
  data_format = _execute.make_str(data_format, "data_format")
  if is_training is None:
    is_training = True
  is_training = _execute.make_bool(is_training, "is_training")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "FusedBatchNormGrad", y_backprop=y_backprop, x=x, scale=scale,
                              reserve_space_1=reserve_space_1,
                              reserve_space_2=reserve_space_2,
                              epsilon=epsilon, data_format=data_format,
                              is_training=is_training, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"), "epsilon",
              _op.get_attr("epsilon"), "data_format",
              _op.get_attr("data_format"), "is_training",
              _op._get_attr_bool("is_training"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "FusedBatchNormGrad", _inputs_flat, _attrs, _result)
  _result = _FusedBatchNormGradOutput._make(_result)
  return _result

FusedBatchNormGrad = tf_export("raw_ops.FusedBatchNormGrad")(_ops.to_raw_op(fused_batch_norm_grad))


def fused_batch_norm_grad_eager_fallback(y_backprop: Annotated[Any, TV_FusedBatchNormGrad_T], x: Annotated[Any, TV_FusedBatchNormGrad_T], scale: Annotated[Any, TV_FusedBatchNormGrad_T], reserve_space_1: Annotated[Any, TV_FusedBatchNormGrad_T], reserve_space_2: Annotated[Any, TV_FusedBatchNormGrad_T], epsilon: float, data_format: str, is_training: bool, name, ctx):
  if epsilon is None:
    epsilon = 0.0001
  epsilon = _execute.make_float(epsilon, "epsilon")
  if data_format is None:
    data_format = "NHWC"
  data_format = _execute.make_str(data_format, "data_format")
  if is_training is None:
    is_training = True
  is_training = _execute.make_bool(is_training, "is_training")
  _attr_T, _inputs_T = _execute.args_to_matching_eager([y_backprop, x, scale, reserve_space_1, reserve_space_2], ctx, [_dtypes.float32, ])
  (y_backprop, x, scale, reserve_space_1, reserve_space_2) = _inputs_T
  _inputs_flat = [y_backprop, x, scale, reserve_space_1, reserve_space_2]
  _attrs = ("T", _attr_T, "epsilon", epsilon, "data_format", data_format,
  "is_training", is_training)
  _result = _execute.execute(b"FusedBatchNormGrad", 5, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "FusedBatchNormGrad", _inputs_flat, _attrs, _result)
  _result = _FusedBatchNormGradOutput._make(_result)
  return _result

_FusedBatchNormGradV2Output = collections.namedtuple(
    "FusedBatchNormGradV2",
    ["x_backprop", "scale_backprop", "offset_backprop", "reserve_space_3", "reserve_space_4"])


TV_FusedBatchNormGradV2_T = TypeVar("TV_FusedBatchNormGradV2_T", _atypes.BFloat16, _atypes.Float32, _atypes.Half)
TV_FusedBatchNormGradV2_U = TypeVar("TV_FusedBatchNormGradV2_U", bound=_atypes.Float32)

def fused_batch_norm_grad_v2(y_backprop: Annotated[Any, TV_FusedBatchNormGradV2_T], x: Annotated[Any, TV_FusedBatchNormGradV2_T], scale: Annotated[Any, _atypes.Float32], reserve_space_1: Annotated[Any, TV_FusedBatchNormGradV2_U], reserve_space_2: Annotated[Any, TV_FusedBatchNormGradV2_U], epsilon:float=0.0001, data_format:str="NHWC", is_training:bool=True, name=None):
  r"""Gradient for batch normalization.

  Note that the size of 4D Tensors are defined by either "NHWC" or "NCHW".
  The size of 1D Tensors matches the dimension C of the 4D Tensors.

  Args:
    y_backprop: A `Tensor`. Must be one of the following types: `half`, `bfloat16`, `float32`.
      A 4D Tensor for the gradient with respect to y.
    x: A `Tensor`. Must have the same type as `y_backprop`.
      A 4D Tensor for input data.
    scale: A `Tensor` of type `float32`.
      A 1D Tensor for scaling factor, to scale the normalized x.
    reserve_space_1: A `Tensor`. Must be one of the following types: `float32`.
      When is_training is True, a 1D Tensor for the computed batch
      mean to be reused in gradient computation. When is_training is
      False, a 1D Tensor for the population mean to be reused in both
      1st and 2nd order gradient computation.
    reserve_space_2: A `Tensor`. Must have the same type as `reserve_space_1`.
      When is_training is True, a 1D Tensor for the computed batch
      variance (inverted variance in the cuDNN case) to be reused in
      gradient computation. When is_training is False, a 1D Tensor
      for the population variance to be reused in both 1st and 2nd
      order gradient computation.
    epsilon: An optional `float`. Defaults to `0.0001`.
      A small float number added to the variance of x.
    data_format: An optional `string` from: `"NHWC", "NCHW"`. Defaults to `"NHWC"`.
      The data format for y_backprop, x, x_backprop.
      Either "NHWC" (default) or "NCHW".
    is_training: An optional `bool`. Defaults to `True`.
      A bool value to indicate the operation is for training (default)
      or inference.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (x_backprop, scale_backprop, offset_backprop, reserve_space_3, reserve_space_4).

    x_backprop: A `Tensor`. Has the same type as `y_backprop`.
    scale_backprop: A `Tensor`. Has the same type as `reserve_space_1`.
    offset_backprop: A `Tensor`. Has the same type as `reserve_space_1`.
    reserve_space_3: A `Tensor`. Has the same type as `reserve_space_1`.
    reserve_space_4: A `Tensor`. Has the same type as `reserve_space_1`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "FusedBatchNormGradV2", name, y_backprop, x, scale,
        reserve_space_1, reserve_space_2, "epsilon", epsilon, "data_format",
        data_format, "is_training", is_training)
      _result = _FusedBatchNormGradV2Output._make(_result)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return fused_batch_norm_grad_v2_eager_fallback(
          y_backprop, x, scale, reserve_space_1, reserve_space_2,
          epsilon=epsilon, data_format=data_format, is_training=is_training,
          name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if epsilon is None:
    epsilon = 0.0001
  epsilon = _execute.make_float(epsilon, "epsilon")
  if data_format is None:
    data_format = "NHWC"
  data_format = _execute.make_str(data_format, "data_format")
  if is_training is None:
    is_training = True
  is_training = _execute.make_bool(is_training, "is_training")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "FusedBatchNormGradV2", y_backprop=y_backprop, x=x, scale=scale,
                                reserve_space_1=reserve_space_1,
                                reserve_space_2=reserve_space_2,
                                epsilon=epsilon, data_format=data_format,
                                is_training=is_training, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"), "U", _op._get_attr_type("U"),
              "epsilon", _op.get_attr("epsilon"), "data_format",
              _op.get_attr("data_format"), "is_training",
              _op._get_attr_bool("is_training"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "FusedBatchNormGradV2", _inputs_flat, _attrs, _result)
  _result = _FusedBatchNormGradV2Output._make(_result)
  return _result

FusedBatchNormGradV2 = tf_export("raw_ops.FusedBatchNormGradV2")(_ops.to_raw_op(fused_batch_norm_grad_v2))


def fused_batch_norm_grad_v2_eager_fallback(y_backprop: Annotated[Any, TV_FusedBatchNormGradV2_T], x: Annotated[Any, TV_FusedBatchNormGradV2_T], scale: Annotated[Any, _atypes.Float32], reserve_space_1: Annotated[Any, TV_FusedBatchNormGradV2_U], reserve_space_2: Annotated[Any, TV_FusedBatchNormGradV2_U], epsilon: float, data_format: str, is_training: bool, name, ctx):
  if epsilon is None:
    epsilon = 0.0001
  epsilon = _execute.make_float(epsilon, "epsilon")
  if data_format is None:
    data_format = "NHWC"
  data_format = _execute.make_str(data_format, "data_format")
  if is_training is None:
    is_training = True
  is_training = _execute.make_bool(is_training, "is_training")
  _attr_T, _inputs_T = _execute.args_to_matching_eager([y_backprop, x], ctx, [_dtypes.half, _dtypes.bfloat16, _dtypes.float32, ])
  (y_backprop, x) = _inputs_T
  _attr_U, _inputs_U = _execute.args_to_matching_eager([reserve_space_1, reserve_space_2], ctx, [_dtypes.float32, ])
  (reserve_space_1, reserve_space_2) = _inputs_U
  scale = _ops.convert_to_tensor(scale, _dtypes.float32)
  _inputs_flat = [y_backprop, x, scale, reserve_space_1, reserve_space_2]
  _attrs = ("T", _attr_T, "U", _attr_U, "epsilon", epsilon, "data_format",
  data_format, "is_training", is_training)
  _result = _execute.execute(b"FusedBatchNormGradV2", 5, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "FusedBatchNormGradV2", _inputs_flat, _attrs, _result)
  _result = _FusedBatchNormGradV2Output._make(_result)
  return _result

_FusedBatchNormGradV3Output = collections.namedtuple(
    "FusedBatchNormGradV3",
    ["x_backprop", "scale_backprop", "offset_backprop", "reserve_space_4", "reserve_space_5"])


TV_FusedBatchNormGradV3_T = TypeVar("TV_FusedBatchNormGradV3_T", _atypes.BFloat16, _atypes.Float32, _atypes.Half)
TV_FusedBatchNormGradV3_U = TypeVar("TV_FusedBatchNormGradV3_U", bound=_atypes.Float32)

def fused_batch_norm_grad_v3(y_backprop: Annotated[Any, TV_FusedBatchNormGradV3_T], x: Annotated[Any, TV_FusedBatchNormGradV3_T], scale: Annotated[Any, _atypes.Float32], reserve_space_1: Annotated[Any, TV_FusedBatchNormGradV3_U], reserve_space_2: Annotated[Any, TV_FusedBatchNormGradV3_U], reserve_space_3: Annotated[Any, TV_FusedBatchNormGradV3_U], epsilon:float=0.0001, data_format:str="NHWC", is_training:bool=True, name=None):
  r"""Gradient for batch normalization.

  Note that the size of 4D Tensors are defined by either "NHWC" or "NCHW".
  The size of 1D Tensors matches the dimension C of the 4D Tensors.

  Args:
    y_backprop: A `Tensor`. Must be one of the following types: `half`, `bfloat16`, `float32`.
      A 4D Tensor for the gradient with respect to y.
    x: A `Tensor`. Must have the same type as `y_backprop`.
      A 4D Tensor for input data.
    scale: A `Tensor` of type `float32`.
      A 1D Tensor for scaling factor, to scale the normalized x.
    reserve_space_1: A `Tensor`. Must be one of the following types: `float32`.
      When is_training is True, a 1D Tensor for the computed batch
      mean to be reused in gradient computation. When is_training is
      False, a 1D Tensor for the population mean to be reused in both
      1st and 2nd order gradient computation.
    reserve_space_2: A `Tensor`. Must have the same type as `reserve_space_1`.
      When is_training is True, a 1D Tensor for the computed batch
      variance (inverted variance in the cuDNN case) to be reused in
      gradient computation. When is_training is False, a 1D Tensor
      for the population variance to be reused in both 1st and 2nd
      order gradient computation.
    reserve_space_3: A `Tensor`. Must have the same type as `reserve_space_1`.
      When is_training is True, a 1D Tensor for some intermediate results to be reused
      in gradient computation. When is_training is False, a dummy empty Tensor will be
      created.
    epsilon: An optional `float`. Defaults to `0.0001`.
      A small float number added to the variance of x.
    data_format: An optional `string` from: `"NHWC", "NCHW", "NDHWC", "NCDHW"`. Defaults to `"NHWC"`.
      The data format for y_backprop, x, x_backprop.
      Either "NHWC" (default) or "NCHW".
    is_training: An optional `bool`. Defaults to `True`.
      A bool value to indicate the operation is for training (default)
      or inference.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (x_backprop, scale_backprop, offset_backprop, reserve_space_4, reserve_space_5).

    x_backprop: A `Tensor`. Has the same type as `y_backprop`.
    scale_backprop: A `Tensor`. Has the same type as `reserve_space_1`.
    offset_backprop: A `Tensor`. Has the same type as `reserve_space_1`.
    reserve_space_4: A `Tensor`. Has the same type as `reserve_space_1`.
    reserve_space_5: A `Tensor`. Has the same type as `reserve_space_1`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "FusedBatchNormGradV3", name, y_backprop, x, scale,
        reserve_space_1, reserve_space_2, reserve_space_3, "epsilon", epsilon,
        "data_format", data_format, "is_training", is_training)
      _result = _FusedBatchNormGradV3Output._make(_result)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return fused_batch_norm_grad_v3_eager_fallback(
          y_backprop, x, scale, reserve_space_1, reserve_space_2,
          reserve_space_3, epsilon=epsilon, data_format=data_format,
          is_training=is_training, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if epsilon is None:
    epsilon = 0.0001
  epsilon = _execute.make_float(epsilon, "epsilon")
  if data_format is None:
    data_format = "NHWC"
  data_format = _execute.make_str(data_format, "data_format")
  if is_training is None:
    is_training = True
  is_training = _execute.make_bool(is_training, "is_training")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "FusedBatchNormGradV3", y_backprop=y_backprop, x=x, scale=scale,
                                reserve_space_1=reserve_space_1,
                                reserve_space_2=reserve_space_2,
                                reserve_space_3=reserve_space_3,
                                epsilon=epsilon, data_format=data_format,
                                is_training=is_training, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"), "U", _op._get_attr_type("U"),
              "epsilon", _op.get_attr("epsilon"), "data_format",
              _op.get_attr("data_format"), "is_training",
              _op._get_attr_bool("is_training"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "FusedBatchNormGradV3", _inputs_flat, _attrs, _result)
  _result = _FusedBatchNormGradV3Output._make(_result)
  return _result

FusedBatchNormGradV3 = tf_export("raw_ops.FusedBatchNormGradV3")(_ops.to_raw_op(fused_batch_norm_grad_v3))


def fused_batch_norm_grad_v3_eager_fallback(y_backprop: Annotated[Any, TV_FusedBatchNormGradV3_T], x: Annotated[Any, TV_FusedBatchNormGradV3_T], scale: Annotated[Any, _atypes.Float32], reserve_space_1: Annotated[Any, TV_FusedBatchNormGradV3_U], reserve_space_2: Annotated[Any, TV_FusedBatchNormGradV3_U], reserve_space_3: Annotated[Any, TV_FusedBatchNormGradV3_U], epsilon: float, data_format: str, is_training: bool, name, ctx):
  if epsilon is None:
    epsilon = 0.0001
  epsilon = _execute.make_float(epsilon, "epsilon")
  if data_format is None:
    data_format = "NHWC"
  data_format = _execute.make_str(data_format, "data_format")
  if is_training is None:
    is_training = True
  is_training = _execute.make_bool(is_training, "is_training")
  _attr_T, _inputs_T = _execute.args_to_matching_eager([y_backprop, x], ctx, [_dtypes.half, _dtypes.bfloat16, _dtypes.float32, ])
  (y_backprop, x) = _inputs_T
  _attr_U, _inputs_U = _execute.args_to_matching_eager([reserve_space_1, reserve_space_2, reserve_space_3], ctx, [_dtypes.float32, ])
  (reserve_space_1, reserve_space_2, reserve_space_3) = _inputs_U
  scale = _ops.convert_to_tensor(scale, _dtypes.float32)
  _inputs_flat = [y_backprop, x, scale, reserve_space_1, reserve_space_2, reserve_space_3]
  _attrs = ("T", _attr_T, "U", _attr_U, "epsilon", epsilon, "data_format",
  data_format, "is_training", is_training)
  _result = _execute.execute(b"FusedBatchNormGradV3", 5, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "FusedBatchNormGradV3", _inputs_flat, _attrs, _result)
  _result = _FusedBatchNormGradV3Output._make(_result)
  return _result

_FusedBatchNormV2Output = collections.namedtuple(
    "FusedBatchNormV2",
    ["y", "batch_mean", "batch_variance", "reserve_space_1", "reserve_space_2"])


TV_FusedBatchNormV2_T = TypeVar("TV_FusedBatchNormV2_T", _atypes.BFloat16, _atypes.Float32, _atypes.Half)
TV_FusedBatchNormV2_U = TypeVar("TV_FusedBatchNormV2_U", bound=_atypes.Float32)

def fused_batch_norm_v2(x: Annotated[Any, TV_FusedBatchNormV2_T], scale: Annotated[Any, TV_FusedBatchNormV2_U], offset: Annotated[Any, TV_FusedBatchNormV2_U], mean: Annotated[Any, TV_FusedBatchNormV2_U], variance: Annotated[Any, TV_FusedBatchNormV2_U], epsilon:float=0.0001, exponential_avg_factor:float=1, data_format:str="NHWC", is_training:bool=True, name=None):
  r"""Batch normalization.

  Note that the size of 4D Tensors are defined by either "NHWC" or "NCHW".
  The size of 1D Tensors matches the dimension C of the 4D Tensors.

  Args:
    x: A `Tensor`. Must be one of the following types: `half`, `bfloat16`, `float32`.
      A 4D Tensor for input data.
    scale: A `Tensor`. Must be one of the following types: `float32`.
      A 1D Tensor for scaling factor, to scale the normalized x.
    offset: A `Tensor`. Must have the same type as `scale`.
      A 1D Tensor for offset, to shift to the normalized x.
    mean: A `Tensor`. Must have the same type as `scale`.
      A 1D Tensor for population mean. Used for inference only;
      must be empty for training.
    variance: A `Tensor`. Must have the same type as `scale`.
      A 1D Tensor for population variance. Used for inference only;
      must be empty for training.
    epsilon: An optional `float`. Defaults to `0.0001`.
      A small float number added to the variance of x.
    exponential_avg_factor: An optional `float`. Defaults to `1`.
    data_format: An optional `string` from: `"NHWC", "NCHW"`. Defaults to `"NHWC"`.
      The data format for x and y. Either "NHWC" (default) or "NCHW".
    is_training: An optional `bool`. Defaults to `True`.
      A bool value to indicate the operation is for training (default)
      or inference.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (y, batch_mean, batch_variance, reserve_space_1, reserve_space_2).

    y: A `Tensor`. Has the same type as `x`.
    batch_mean: A `Tensor`. Has the same type as `scale`.
    batch_variance: A `Tensor`. Has the same type as `scale`.
    reserve_space_1: A `Tensor`. Has the same type as `scale`.
    reserve_space_2: A `Tensor`. Has the same type as `scale`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "FusedBatchNormV2", name, x, scale, offset, mean, variance,
        "epsilon", epsilon, "exponential_avg_factor", exponential_avg_factor,
        "data_format", data_format, "is_training", is_training)
      _result = _FusedBatchNormV2Output._make(_result)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return fused_batch_norm_v2_eager_fallback(
          x, scale, offset, mean, variance, epsilon=epsilon,
          exponential_avg_factor=exponential_avg_factor,
          data_format=data_format, is_training=is_training, name=name,
          ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if epsilon is None:
    epsilon = 0.0001
  epsilon = _execute.make_float(epsilon, "epsilon")
  if exponential_avg_factor is None:
    exponential_avg_factor = 1
  exponential_avg_factor = _execute.make_float(exponential_avg_factor, "exponential_avg_factor")
  if data_format is None:
    data_format = "NHWC"
  data_format = _execute.make_str(data_format, "data_format")
  if is_training is None:
    is_training = True
  is_training = _execute.make_bool(is_training, "is_training")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "FusedBatchNormV2", x=x, scale=scale, offset=offset, mean=mean,
                            variance=variance, epsilon=epsilon,
                            exponential_avg_factor=exponential_avg_factor,
                            data_format=data_format, is_training=is_training,
                            name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"), "U", _op._get_attr_type("U"),
              "epsilon", _op.get_attr("epsilon"), "exponential_avg_factor",
              _op.get_attr("exponential_avg_factor"), "data_format",
              _op.get_attr("data_format"), "is_training",
              _op._get_attr_bool("is_training"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "FusedBatchNormV2", _inputs_flat, _attrs, _result)
  _result = _FusedBatchNormV2Output._make(_result)
  return _result

FusedBatchNormV2 = tf_export("raw_ops.FusedBatchNormV2")(_ops.to_raw_op(fused_batch_norm_v2))


def fused_batch_norm_v2_eager_fallback(x: Annotated[Any, TV_FusedBatchNormV2_T], scale: Annotated[Any, TV_FusedBatchNormV2_U], offset: Annotated[Any, TV_FusedBatchNormV2_U], mean: Annotated[Any, TV_FusedBatchNormV2_U], variance: Annotated[Any, TV_FusedBatchNormV2_U], epsilon: float, exponential_avg_factor: float, data_format: str, is_training: bool, name, ctx):
  if epsilon is None:
    epsilon = 0.0001
  epsilon = _execute.make_float(epsilon, "epsilon")
  if exponential_avg_factor is None:
    exponential_avg_factor = 1
  exponential_avg_factor = _execute.make_float(exponential_avg_factor, "exponential_avg_factor")
  if data_format is None:
    data_format = "NHWC"
  data_format = _execute.make_str(data_format, "data_format")
  if is_training is None:
    is_training = True
  is_training = _execute.make_bool(is_training, "is_training")
  _attr_T, (x,) = _execute.args_to_matching_eager([x], ctx, [_dtypes.half, _dtypes.bfloat16, _dtypes.float32, ])
  _attr_U, _inputs_U = _execute.args_to_matching_eager([scale, offset, mean, variance], ctx, [_dtypes.float32, ])
  (scale, offset, mean, variance) = _inputs_U
  _inputs_flat = [x, scale, offset, mean, variance]
  _attrs = ("T", _attr_T, "U", _attr_U, "epsilon", epsilon,
  "exponential_avg_factor", exponential_avg_factor, "data_format",
  data_format, "is_training", is_training)
  _result = _execute.execute(b"FusedBatchNormV2", 5, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "FusedBatchNormV2", _inputs_flat, _attrs, _result)
  _result = _FusedBatchNormV2Output._make(_result)
  return _result

_FusedBatchNormV3Output = collections.namedtuple(
    "FusedBatchNormV3",
    ["y", "batch_mean", "batch_variance", "reserve_space_1", "reserve_space_2", "reserve_space_3"])


TV_FusedBatchNormV3_T = TypeVar("TV_FusedBatchNormV3_T", _atypes.BFloat16, _atypes.Float32, _atypes.Half)
TV_FusedBatchNormV3_U = TypeVar("TV_FusedBatchNormV3_U", _atypes.BFloat16, _atypes.Float32)

def fused_batch_norm_v3(x: Annotated[Any, TV_FusedBatchNormV3_T], scale: Annotated[Any, TV_FusedBatchNormV3_U], offset: Annotated[Any, TV_FusedBatchNormV3_U], mean: Annotated[Any, TV_FusedBatchNormV3_U], variance: Annotated[Any, TV_FusedBatchNormV3_U], epsilon:float=0.0001, exponential_avg_factor:float=1, data_format:str="NHWC", is_training:bool=True, name=None):
  r"""Batch normalization.

  Note that the size of 4D Tensors are defined by either "NHWC" or "NCHW".
  The size of 1D Tensors matches the dimension C of the 4D Tensors.

  Args:
    x: A `Tensor`. Must be one of the following types: `half`, `bfloat16`, `float32`.
      A 4D Tensor for input data.
    scale: A `Tensor`. Must be one of the following types: `bfloat16`, `float32`.
      A 1D Tensor for scaling factor, to scale the normalized x.
    offset: A `Tensor`. Must have the same type as `scale`.
      A 1D Tensor for offset, to shift to the normalized x.
    mean: A `Tensor`. Must have the same type as `scale`.
      A 1D Tensor for population mean. Used for inference only;
      must be empty for training.
    variance: A `Tensor`. Must have the same type as `scale`.
      A 1D Tensor for population variance. Used for inference only;
      must be empty for training.
    epsilon: An optional `float`. Defaults to `0.0001`.
      A small float number added to the variance of x.
    exponential_avg_factor: An optional `float`. Defaults to `1`.
    data_format: An optional `string` from: `"NHWC", "NCHW", "NDHWC", "NCDHW"`. Defaults to `"NHWC"`.
      The data format for x and y. Either "NHWC" (default) or "NCHW".
    is_training: An optional `bool`. Defaults to `True`.
      A bool value to indicate the operation is for training (default)
      or inference.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (y, batch_mean, batch_variance, reserve_space_1, reserve_space_2, reserve_space_3).

    y: A `Tensor`. Has the same type as `x`.
    batch_mean: A `Tensor`. Has the same type as `scale`.
    batch_variance: A `Tensor`. Has the same type as `scale`.
    reserve_space_1: A `Tensor`. Has the same type as `scale`.
    reserve_space_2: A `Tensor`. Has the same type as `scale`.
    reserve_space_3: A `Tensor`. Has the same type as `scale`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "FusedBatchNormV3", name, x, scale, offset, mean, variance,
        "epsilon", epsilon, "exponential_avg_factor", exponential_avg_factor,
        "data_format", data_format, "is_training", is_training)
      _result = _FusedBatchNormV3Output._make(_result)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return fused_batch_norm_v3_eager_fallback(
          x, scale, offset, mean, variance, epsilon=epsilon,
          exponential_avg_factor=exponential_avg_factor,
          data_format=data_format, is_training=is_training, name=name,
          ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if epsilon is None:
    epsilon = 0.0001
  epsilon = _execute.make_float(epsilon, "epsilon")
  if exponential_avg_factor is None:
    exponential_avg_factor = 1
  exponential_avg_factor = _execute.make_float(exponential_avg_factor, "exponential_avg_factor")
  if data_format is None:
    data_format = "NHWC"
  data_format = _execute.make_str(data_format, "data_format")
  if is_training is None:
    is_training = True
  is_training = _execute.make_bool(is_training, "is_training")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "FusedBatchNormV3", x=x, scale=scale, offset=offset, mean=mean,
                            variance=variance, epsilon=epsilon,
                            exponential_avg_factor=exponential_avg_factor,
                            data_format=data_format, is_training=is_training,
                            name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"), "U", _op._get_attr_type("U"),
              "epsilon", _op.get_attr("epsilon"), "exponential_avg_factor",
              _op.get_attr("exponential_avg_factor"), "data_format",
              _op.get_attr("data_format"), "is_training",
              _op._get_attr_bool("is_training"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "FusedBatchNormV3", _inputs_flat, _attrs, _result)
  _result = _FusedBatchNormV3Output._make(_result)
  return _result

FusedBatchNormV3 = tf_export("raw_ops.FusedBatchNormV3")(_ops.to_raw_op(fused_batch_norm_v3))


def fused_batch_norm_v3_eager_fallback(x: Annotated[Any, TV_FusedBatchNormV3_T], scale: Annotated[Any, TV_FusedBatchNormV3_U], offset: Annotated[Any, TV_FusedBatchNormV3_U], mean: Annotated[Any, TV_FusedBatchNormV3_U], variance: Annotated[Any, TV_FusedBatchNormV3_U], epsilon: float, exponential_avg_factor: float, data_format: str, is_training: bool, name, ctx):
  if epsilon is None:
    epsilon = 0.0001
  epsilon = _execute.make_float(epsilon, "epsilon")
  if exponential_avg_factor is None:
    exponential_avg_factor = 1
  exponential_avg_factor = _execute.make_float(exponential_avg_factor, "exponential_avg_factor")
  if data_format is None:
    data_format = "NHWC"
  data_format = _execute.make_str(data_format, "data_format")
  if is_training is None:
    is_training = True
  is_training = _execute.make_bool(is_training, "is_training")
  _attr_T, (x,) = _execute.args_to_matching_eager([x], ctx, [_dtypes.half, _dtypes.bfloat16, _dtypes.float32, ])
  _attr_U, _inputs_U = _execute.args_to_matching_eager([scale, offset, mean, variance], ctx, [_dtypes.bfloat16, _dtypes.float32, ])
  (scale, offset, mean, variance) = _inputs_U
  _inputs_flat = [x, scale, offset, mean, variance]
  _attrs = ("T", _attr_T, "U", _attr_U, "epsilon", epsilon,
  "exponential_avg_factor", exponential_avg_factor, "data_format",
  data_format, "is_training", is_training)
  _result = _execute.execute(b"FusedBatchNormV3", 6, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "FusedBatchNormV3", _inputs_flat, _attrs, _result)
  _result = _FusedBatchNormV3Output._make(_result)
  return _result


TV_FusedPadConv2D_T = TypeVar("TV_FusedPadConv2D_T", _atypes.BFloat16, _atypes.Float32, _atypes.Float64, _atypes.Half)

def fused_pad_conv2d(input: Annotated[Any, TV_FusedPadConv2D_T], paddings: Annotated[Any, _atypes.Int32], filter: Annotated[Any, TV_FusedPadConv2D_T], mode: str, strides, padding: str, name=None) -> Annotated[Any, TV_FusedPadConv2D_T]:
  r"""Performs a padding as a preprocess during a convolution.

  Similar to FusedResizeAndPadConv2d, this op allows for an optimized
  implementation where the spatial padding transformation stage is fused with the
  im2col lookup, but in this case without the bilinear filtering required for
  resizing. Fusing the padding prevents the need to write out the intermediate
  results as whole tensors, reducing memory pressure, and we can get some latency
  gains by merging the transformation calculations.
  The data_format attribute for Conv2D isn't supported by this op, and 'NHWC'
  order is used instead.
  Internally this op uses a single per-graph scratch buffer, which means that it
  will block if multiple versions are being run in parallel. This is because this
  operator is primarily an optimization to minimize memory usage.

  Args:
    input: A `Tensor`. Must be one of the following types: `half`, `bfloat16`, `float32`, `float64`.
      4-D with shape `[batch, in_height, in_width, in_channels]`.
    paddings: A `Tensor` of type `int32`.
      A two-column matrix specifying the padding sizes. The number of
      rows must be the same as the rank of `input`.
    filter: A `Tensor`. Must have the same type as `input`. 4-D with shape
      `[filter_height, filter_width, in_channels, out_channels]`.
    mode: A `string` from: `"REFLECT", "SYMMETRIC"`.
    strides: A list of `ints`.
      1-D of length 4.  The stride of the sliding window for each dimension
      of `input`. Must be in the same order as the dimension specified with format.
    padding: A `string` from: `"SAME", "VALID"`.
      The type of padding algorithm to use.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "FusedPadConv2D", name, input, paddings, filter, "mode", mode,
        "strides", strides, "padding", padding)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return fused_pad_conv2d_eager_fallback(
          input, paddings, filter, mode=mode, strides=strides,
          padding=padding, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  mode = _execute.make_str(mode, "mode")
  if not isinstance(strides, (list, tuple)):
    raise TypeError(
        "Expected list for 'strides' argument to "
        "'fused_pad_conv2d' Op, not %r." % strides)
  strides = [_execute.make_int(_i, "strides") for _i in strides]
  padding = _execute.make_str(padding, "padding")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "FusedPadConv2D", input=input, paddings=paddings, filter=filter,
                          mode=mode, strides=strides, padding=padding,
                          name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"), "mode", _op.get_attr("mode"),
              "strides", _op.get_attr("strides"), "padding",
              _op.get_attr("padding"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "FusedPadConv2D", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

FusedPadConv2D = tf_export("raw_ops.FusedPadConv2D")(_ops.to_raw_op(fused_pad_conv2d))


def fused_pad_conv2d_eager_fallback(input: Annotated[Any, TV_FusedPadConv2D_T], paddings: Annotated[Any, _atypes.Int32], filter: Annotated[Any, TV_FusedPadConv2D_T], mode: str, strides, padding: str, name, ctx) -> Annotated[Any, TV_FusedPadConv2D_T]:
  mode = _execute.make_str(mode, "mode")
  if not isinstance(strides, (list, tuple)):
    raise TypeError(
        "Expected list for 'strides' argument to "
        "'fused_pad_conv2d' Op, not %r." % strides)
  strides = [_execute.make_int(_i, "strides") for _i in strides]
  padding = _execute.make_str(padding, "padding")
  _attr_T, _inputs_T = _execute.args_to_matching_eager([input, filter], ctx, [_dtypes.half, _dtypes.bfloat16, _dtypes.float32, _dtypes.float64, ])
  (input, filter) = _inputs_T
  paddings = _ops.convert_to_tensor(paddings, _dtypes.int32)
  _inputs_flat = [input, paddings, filter]
  _attrs = ("T", _attr_T, "mode", mode, "strides", strides, "padding",
  padding)
  _result = _execute.execute(b"FusedPadConv2D", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "FusedPadConv2D", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_FusedResizeAndPadConv2D_T = TypeVar("TV_FusedResizeAndPadConv2D_T", _atypes.Float32, _atypes.Float64, _atypes.Half)

def fused_resize_and_pad_conv2d(input: Annotated[Any, TV_FusedResizeAndPadConv2D_T], size: Annotated[Any, _atypes.Int32], paddings: Annotated[Any, _atypes.Int32], filter: Annotated[Any, TV_FusedResizeAndPadConv2D_T], mode: str, strides, padding: str, resize_align_corners:bool=False, name=None) -> Annotated[Any, TV_FusedResizeAndPadConv2D_T]:
  r"""Performs a resize and padding as a preprocess during a convolution.

  It's often possible to do spatial transformations more efficiently as part of
  the packing stage of a convolution, so this op allows for an optimized
  implementation where these stages are fused together. This prevents the need to
  write out the intermediate results as whole tensors, reducing memory pressure,
  and we can get some latency gains by merging the transformation calculations.
  The data_format attribute for Conv2D isn't supported by this op, and defaults to
  'NHWC' order.
  Internally this op uses a single per-graph scratch buffer, which means that it
  will block if multiple versions are being run in parallel. This is because this
  operator is primarily an optimization to minimize memory usage.

  Args:
    input: A `Tensor`. Must be one of the following types: `half`, `float32`, `float64`.
      4-D with shape `[batch, in_height, in_width, in_channels]`.
    size: A `Tensor` of type `int32`.
      A 1-D int32 Tensor of 2 elements: `new_height, new_width`.  The
      new size for the images.
    paddings: A `Tensor` of type `int32`.
      A two-column matrix specifying the padding sizes. The number of
      rows must be the same as the rank of `input`.
    filter: A `Tensor`. Must have the same type as `input`. 4-D with shape
      `[filter_height, filter_width, in_channels, out_channels]`.
    mode: A `string` from: `"REFLECT", "SYMMETRIC"`.
    strides: A list of `ints`.
      1-D of length 4.  The stride of the sliding window for each dimension
      of `input`. Must be in the same order as the dimension specified with format.
    padding: A `string` from: `"SAME", "VALID"`.
      The type of padding algorithm to use.
    resize_align_corners: An optional `bool`. Defaults to `False`.
      If true, the centers of the 4 corner pixels of the input and output tensors are
      aligned, preserving the values at the corner pixels. Defaults to false.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "FusedResizeAndPadConv2D", name, input, size, paddings, filter,
        "resize_align_corners", resize_align_corners, "mode", mode, "strides",
        strides, "padding", padding)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return fused_resize_and_pad_conv2d_eager_fallback(
          input, size, paddings, filter,
          resize_align_corners=resize_align_corners, mode=mode,
          strides=strides, padding=padding, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  mode = _execute.make_str(mode, "mode")
  if not isinstance(strides, (list, tuple)):
    raise TypeError(
        "Expected list for 'strides' argument to "
        "'fused_resize_and_pad_conv2d' Op, not %r." % strides)
  strides = [_execute.make_int(_i, "strides") for _i in strides]
  padding = _execute.make_str(padding, "padding")
  if resize_align_corners is None:
    resize_align_corners = False
  resize_align_corners = _execute.make_bool(resize_align_corners, "resize_align_corners")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "FusedResizeAndPadConv2D", input=input, size=size, paddings=paddings,
                                   filter=filter, mode=mode, strides=strides,
                                   padding=padding,
                                   resize_align_corners=resize_align_corners,
                                   name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"), "resize_align_corners",
              _op._get_attr_bool("resize_align_corners"), "mode",
              _op.get_attr("mode"), "strides", _op.get_attr("strides"),
              "padding", _op.get_attr("padding"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "FusedResizeAndPadConv2D", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

FusedResizeAndPadConv2D = tf_export("raw_ops.FusedResizeAndPadConv2D")(_ops.to_raw_op(fused_resize_and_pad_conv2d))


def fused_resize_and_pad_conv2d_eager_fallback(input: Annotated[Any, TV_FusedResizeAndPadConv2D_T], size: Annotated[Any, _atypes.Int32], paddings: Annotated[Any, _atypes.Int32], filter: Annotated[Any, TV_FusedResizeAndPadConv2D_T], mode: str, strides, padding: str, resize_align_corners: bool, name, ctx) -> Annotated[Any, TV_FusedResizeAndPadConv2D_T]:
  mode = _execute.make_str(mode, "mode")
  if not isinstance(strides, (list, tuple)):
    raise TypeError(
        "Expected list for 'strides' argument to "
        "'fused_resize_and_pad_conv2d' Op, not %r." % strides)
  strides = [_execute.make_int(_i, "strides") for _i in strides]
  padding = _execute.make_str(padding, "padding")
  if resize_align_corners is None:
    resize_align_corners = False
  resize_align_corners = _execute.make_bool(resize_align_corners, "resize_align_corners")
  _attr_T, _inputs_T = _execute.args_to_matching_eager([input, filter], ctx, [_dtypes.half, _dtypes.float32, _dtypes.float64, ])
  (input, filter) = _inputs_T
  size = _ops.convert_to_tensor(size, _dtypes.int32)
  paddings = _ops.convert_to_tensor(paddings, _dtypes.int32)
  _inputs_flat = [input, size, paddings, filter]
  _attrs = ("T", _attr_T, "resize_align_corners", resize_align_corners,
  "mode", mode, "strides", strides, "padding", padding)
  _result = _execute.execute(b"FusedResizeAndPadConv2D", 1,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "FusedResizeAndPadConv2D", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_InTopK_T = TypeVar("TV_InTopK_T", _atypes.Int32, _atypes.Int64)

def in_top_k(predictions: Annotated[Any, _atypes.Float32], targets: Annotated[Any, TV_InTopK_T], k: int, name=None) -> Annotated[Any, _atypes.Bool]:
  r"""Says whether the targets are in the top `K` predictions.

  This outputs a `batch_size` bool array, an entry `out[i]` is `true` if the
  prediction for the target class is among the top `k` predictions among
  all predictions for example `i`. Note that the behavior of `InTopK` differs
  from the `TopK` op in its handling of ties; if multiple classes have the
  same prediction value and straddle the top-`k` boundary, all of those
  classes are considered to be in the top `k`.

  More formally, let

    \\(predictions_i\\) be the predictions for all classes for example `i`,
    \\(targets_i\\) be the target class for example `i`,
    \\(out_i\\) be the output for example `i`,

  $$out_i = predictions_{i, targets_i} \in TopKIncludingTies(predictions_i)$$

  Args:
    predictions: A `Tensor` of type `float32`.
      A `batch_size` x `classes` tensor.
    targets: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      A `batch_size` vector of class ids.
    k: An `int`. Number of top elements to look at for computing precision.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `bool`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "InTopK", name, predictions, targets, "k", k)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return in_top_k_eager_fallback(
          predictions, targets, k=k, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  k = _execute.make_int(k, "k")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "InTopK", predictions=predictions, targets=targets, k=k, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("k", _op._get_attr_int("k"), "T", _op._get_attr_type("T"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "InTopK", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

InTopK = tf_export("raw_ops.InTopK")(_ops.to_raw_op(in_top_k))


def in_top_k_eager_fallback(predictions: Annotated[Any, _atypes.Float32], targets: Annotated[Any, TV_InTopK_T], k: int, name, ctx) -> Annotated[Any, _atypes.Bool]:
  k = _execute.make_int(k, "k")
  _attr_T, (targets,) = _execute.args_to_matching_eager([targets], ctx, [_dtypes.int32, _dtypes.int64, ], _dtypes.int32)
  predictions = _ops.convert_to_tensor(predictions, _dtypes.float32)
  _inputs_flat = [predictions, targets]
  _attrs = ("k", k, "T", _attr_T)
  _result = _execute.execute(b"InTopK", 1, inputs=_inputs_flat, attrs=_attrs,
                             ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "InTopK", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_InTopKV2_T = TypeVar("TV_InTopKV2_T", _atypes.Int32, _atypes.Int64)

def in_top_kv2(predictions: Annotated[Any, _atypes.Float32], targets: Annotated[Any, TV_InTopKV2_T], k: Annotated[Any, TV_InTopKV2_T], name=None) -> Annotated[Any, _atypes.Bool]:
  r"""Says whether the targets are in the top `K` predictions.

  This outputs a `batch_size` bool array, an entry `out[i]` is `true` if the
  prediction for the target class is among the top `k` predictions among
  all predictions for example `i`. Note that the behavior of `InTopK` differs
  from the `TopK` op in its handling of ties; if multiple classes have the
  same prediction value and straddle the top-`k` boundary, all of those
  classes are considered to be in the top `k`.

  More formally, let

    \\(predictions_i\\) be the predictions for all classes for example `i`,
    \\(targets_i\\) be the target class for example `i`,
    \\(out_i\\) be the output for example `i`,

  $$out_i = predictions_{i, targets_i} \in TopKIncludingTies(predictions_i)$$

  Args:
    predictions: A `Tensor` of type `float32`.
      A `batch_size` x `classes` tensor.
    targets: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      A `batch_size` vector of class ids.
    k: A `Tensor`. Must have the same type as `targets`.
      Number of top elements to look at for computing precision.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `bool`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "InTopKV2", name, predictions, targets, k)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return in_top_kv2_eager_fallback(
          predictions, targets, k, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "InTopKV2", predictions=predictions, targets=targets, k=k, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "InTopKV2", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

InTopKV2 = tf_export("raw_ops.InTopKV2")(_ops.to_raw_op(in_top_kv2))


def in_top_kv2_eager_fallback(predictions: Annotated[Any, _atypes.Float32], targets: Annotated[Any, TV_InTopKV2_T], k: Annotated[Any, TV_InTopKV2_T], name, ctx) -> Annotated[Any, _atypes.Bool]:
  _attr_T, _inputs_T = _execute.args_to_matching_eager([targets, k], ctx, [_dtypes.int32, _dtypes.int64, ], _dtypes.int32)
  (targets, k) = _inputs_T
  predictions = _ops.convert_to_tensor(predictions, _dtypes.float32)
  _inputs_flat = [predictions, targets, k]
  _attrs = ("T", _attr_T)
  _result = _execute.execute(b"InTopKV2", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "InTopKV2", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

_IsotonicRegressionOutput = collections.namedtuple(
    "IsotonicRegression",
    ["output", "segments"])


TV_IsotonicRegression_T = TypeVar("TV_IsotonicRegression_T", _atypes.BFloat16, _atypes.Float32, _atypes.Float64, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int64, _atypes.Int8, _atypes.UInt16, _atypes.UInt32, _atypes.UInt64, _atypes.UInt8)
TV_IsotonicRegression_output_dtype = TypeVar("TV_IsotonicRegression_output_dtype", _atypes.BFloat16, _atypes.Float32, _atypes.Float64, _atypes.Half)

def isotonic_regression(input: Annotated[Any, TV_IsotonicRegression_T], output_dtype:TV_IsotonicRegression_output_dtype=_dtypes.float32, name=None):
  r"""Solves a batch of isotonic regression problems.

  Args:
    input: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`, `int64`, `bfloat16`, `uint16`, `half`, `uint32`, `uint64`.
      A (batch_size, dim)-tensor holding a batch of inputs.
    output_dtype: An optional `tf.DType` from: `tf.half, tf.bfloat16, tf.float32, tf.float64`. Defaults to `tf.float32`.
      Dtype of output.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (output, segments).

    output: A `Tensor` of type `output_dtype`.
    segments: A `Tensor` of type `int32`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "IsotonicRegression", name, input, "output_dtype", output_dtype)
      _result = _IsotonicRegressionOutput._make(_result)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return isotonic_regression_eager_fallback(
          input, output_dtype=output_dtype, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if output_dtype is None:
    output_dtype = _dtypes.float32
  output_dtype = _execute.make_type(output_dtype, "output_dtype")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "IsotonicRegression", input=input, output_dtype=output_dtype,
                              name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"), "output_dtype",
              _op._get_attr_type("output_dtype"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "IsotonicRegression", _inputs_flat, _attrs, _result)
  _result = _IsotonicRegressionOutput._make(_result)
  return _result

IsotonicRegression = tf_export("raw_ops.IsotonicRegression")(_ops.to_raw_op(isotonic_regression))


def isotonic_regression_eager_fallback(input: Annotated[Any, TV_IsotonicRegression_T], output_dtype: TV_IsotonicRegression_output_dtype, name, ctx):
  if output_dtype is None:
    output_dtype = _dtypes.float32
  output_dtype = _execute.make_type(output_dtype, "output_dtype")
  _attr_T, (input,) = _execute.args_to_matching_eager([input], ctx, [_dtypes.float32, _dtypes.float64, _dtypes.int32, _dtypes.uint8, _dtypes.int16, _dtypes.int8, _dtypes.int64, _dtypes.bfloat16, _dtypes.uint16, _dtypes.half, _dtypes.uint32, _dtypes.uint64, ])
  _inputs_flat = [input]
  _attrs = ("T", _attr_T, "output_dtype", output_dtype)
  _result = _execute.execute(b"IsotonicRegression", 2, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "IsotonicRegression", _inputs_flat, _attrs, _result)
  _result = _IsotonicRegressionOutput._make(_result)
  return _result


TV_L2Loss_T = TypeVar("TV_L2Loss_T", _atypes.BFloat16, _atypes.Float32, _atypes.Float64, _atypes.Half)

@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('nn.l2_loss')
def l2_loss(t: Annotated[Any, TV_L2Loss_T], name=None) -> Annotated[Any, TV_L2Loss_T]:
  r"""L2 Loss.

  Computes half the L2 norm of a tensor without the `sqrt`:

      output = sum(t ** 2) / 2

  Args:
    t: A `Tensor`. Must be one of the following types: `half`, `bfloat16`, `float32`, `float64`.
      Typically 2-D, but may have any dimensions.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `t`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "L2Loss", name, t)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_l2_loss(
          (t, name,), None)
      if _result is not NotImplemented:
        return _result
      return l2_loss_eager_fallback(
          t, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            l2_loss, (), dict(t=t, name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_l2_loss(
        (t, name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "L2Loss", t=t, name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          l2_loss, (), dict(t=t, name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "L2Loss", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

L2Loss = tf_export("raw_ops.L2Loss")(_ops.to_raw_op(l2_loss))
_dispatcher_for_l2_loss = l2_loss._tf_type_based_dispatcher.Dispatch


def l2_loss_eager_fallback(t: Annotated[Any, TV_L2Loss_T], name, ctx) -> Annotated[Any, TV_L2Loss_T]:
  _attr_T, (t,) = _execute.args_to_matching_eager([t], ctx, [_dtypes.half, _dtypes.bfloat16, _dtypes.float32, _dtypes.float64, ])
  _inputs_flat = [t]
  _attrs = ("T", _attr_T)
  _result = _execute.execute(b"L2Loss", 1, inputs=_inputs_flat, attrs=_attrs,
                             ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "L2Loss", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_LRN_T = TypeVar("TV_LRN_T", _atypes.BFloat16, _atypes.Float32, _atypes.Half)

@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('nn.local_response_normalization', 'nn.lrn')
def lrn(input: Annotated[Any, TV_LRN_T], depth_radius:int=5, bias:float=1, alpha:float=1, beta:float=0.5, name=None) -> Annotated[Any, TV_LRN_T]:
  r"""Local Response Normalization.

  The 4-D `input` tensor is treated as a 3-D array of 1-D vectors (along the last
  dimension), and each vector is normalized independently.  Within a given vector,
  each component is divided by the weighted, squared sum of inputs within
  `depth_radius`.  In detail,

      sqr_sum[a, b, c, d] =
          sum(input[a, b, c, d - depth_radius : d + depth_radius + 1] ** 2)
      output = input / (bias + alpha * sqr_sum) ** beta

  For details, see [Krizhevsky et al., ImageNet classification with deep
  convolutional neural networks (NIPS 2012)](http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks).

  Args:
    input: A `Tensor`. Must be one of the following types: `half`, `bfloat16`, `float32`.
      4-D.
    depth_radius: An optional `int`. Defaults to `5`.
      0-D.  Half-width of the 1-D normalization window.
    bias: An optional `float`. Defaults to `1`.
      An offset (usually positive to avoid dividing by 0).
    alpha: An optional `float`. Defaults to `1`.
      A scale factor, usually positive.
    beta: An optional `float`. Defaults to `0.5`. An exponent.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "LRN", name, input, "depth_radius", depth_radius, "bias", bias,
        "alpha", alpha, "beta", beta)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_lrn(
          (input, depth_radius, bias, alpha, beta, name,), None)
      if _result is not NotImplemented:
        return _result
      return lrn_eager_fallback(
          input, depth_radius=depth_radius, bias=bias, alpha=alpha, beta=beta,
          name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            lrn, (), dict(input=input, depth_radius=depth_radius, bias=bias,
                          alpha=alpha, beta=beta, name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_lrn(
        (input, depth_radius, bias, alpha, beta, name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  if depth_radius is None:
    depth_radius = 5
  depth_radius = _execute.make_int(depth_radius, "depth_radius")
  if bias is None:
    bias = 1
  bias = _execute.make_float(bias, "bias")
  if alpha is None:
    alpha = 1
  alpha = _execute.make_float(alpha, "alpha")
  if beta is None:
    beta = 0.5
  beta = _execute.make_float(beta, "beta")
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "LRN", input=input, depth_radius=depth_radius, bias=bias, alpha=alpha,
               beta=beta, name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          lrn, (), dict(input=input, depth_radius=depth_radius, bias=bias,
                        alpha=alpha, beta=beta, name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("depth_radius", _op._get_attr_int("depth_radius"), "bias",
              _op.get_attr("bias"), "alpha", _op.get_attr("alpha"), "beta",
              _op.get_attr("beta"), "T", _op._get_attr_type("T"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "LRN", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

LRN = tf_export("raw_ops.LRN")(_ops.to_raw_op(lrn))
_dispatcher_for_lrn = lrn._tf_type_based_dispatcher.Dispatch


def lrn_eager_fallback(input: Annotated[Any, TV_LRN_T], depth_radius: int, bias: float, alpha: float, beta: float, name, ctx) -> Annotated[Any, TV_LRN_T]:
  if depth_radius is None:
    depth_radius = 5
  depth_radius = _execute.make_int(depth_radius, "depth_radius")
  if bias is None:
    bias = 1
  bias = _execute.make_float(bias, "bias")
  if alpha is None:
    alpha = 1
  alpha = _execute.make_float(alpha, "alpha")
  if beta is None:
    beta = 0.5
  beta = _execute.make_float(beta, "beta")
  _attr_T, (input,) = _execute.args_to_matching_eager([input], ctx, [_dtypes.half, _dtypes.bfloat16, _dtypes.float32, ], _dtypes.float32)
  _inputs_flat = [input]
  _attrs = ("depth_radius", depth_radius, "bias", bias, "alpha", alpha,
  "beta", beta, "T", _attr_T)
  _result = _execute.execute(b"LRN", 1, inputs=_inputs_flat, attrs=_attrs,
                             ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "LRN", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_LRNGrad_T = TypeVar("TV_LRNGrad_T", _atypes.BFloat16, _atypes.Float32, _atypes.Half)

def lrn_grad(input_grads: Annotated[Any, TV_LRNGrad_T], input_image: Annotated[Any, TV_LRNGrad_T], output_image: Annotated[Any, TV_LRNGrad_T], depth_radius:int=5, bias:float=1, alpha:float=1, beta:float=0.5, name=None) -> Annotated[Any, TV_LRNGrad_T]:
  r"""Gradients for Local Response Normalization.

  Args:
    input_grads: A `Tensor`. Must be one of the following types: `half`, `bfloat16`, `float32`.
      4-D with shape `[batch, height, width, channels]`.
    input_image: A `Tensor`. Must have the same type as `input_grads`.
      4-D with shape `[batch, height, width, channels]`.
    output_image: A `Tensor`. Must have the same type as `input_grads`.
      4-D with shape `[batch, height, width, channels]`.
    depth_radius: An optional `int`. Defaults to `5`. A depth radius.
    bias: An optional `float`. Defaults to `1`.
      An offset (usually > 0 to avoid dividing by 0).
    alpha: An optional `float`. Defaults to `1`.
      A scale factor, usually positive.
    beta: An optional `float`. Defaults to `0.5`. An exponent.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input_grads`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "LRNGrad", name, input_grads, input_image, output_image,
        "depth_radius", depth_radius, "bias", bias, "alpha", alpha, "beta",
        beta)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return lrn_grad_eager_fallback(
          input_grads, input_image, output_image, depth_radius=depth_radius,
          bias=bias, alpha=alpha, beta=beta, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if depth_radius is None:
    depth_radius = 5
  depth_radius = _execute.make_int(depth_radius, "depth_radius")
  if bias is None:
    bias = 1
  bias = _execute.make_float(bias, "bias")
  if alpha is None:
    alpha = 1
  alpha = _execute.make_float(alpha, "alpha")
  if beta is None:
    beta = 0.5
  beta = _execute.make_float(beta, "beta")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "LRNGrad", input_grads=input_grads, input_image=input_image,
                   output_image=output_image, depth_radius=depth_radius,
                   bias=bias, alpha=alpha, beta=beta, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("depth_radius", _op._get_attr_int("depth_radius"), "bias",
              _op.get_attr("bias"), "alpha", _op.get_attr("alpha"), "beta",
              _op.get_attr("beta"), "T", _op._get_attr_type("T"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "LRNGrad", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

LRNGrad = tf_export("raw_ops.LRNGrad")(_ops.to_raw_op(lrn_grad))


def lrn_grad_eager_fallback(input_grads: Annotated[Any, TV_LRNGrad_T], input_image: Annotated[Any, TV_LRNGrad_T], output_image: Annotated[Any, TV_LRNGrad_T], depth_radius: int, bias: float, alpha: float, beta: float, name, ctx) -> Annotated[Any, TV_LRNGrad_T]:
  if depth_radius is None:
    depth_radius = 5
  depth_radius = _execute.make_int(depth_radius, "depth_radius")
  if bias is None:
    bias = 1
  bias = _execute.make_float(bias, "bias")
  if alpha is None:
    alpha = 1
  alpha = _execute.make_float(alpha, "alpha")
  if beta is None:
    beta = 0.5
  beta = _execute.make_float(beta, "beta")
  _attr_T, _inputs_T = _execute.args_to_matching_eager([input_grads, input_image, output_image], ctx, [_dtypes.half, _dtypes.bfloat16, _dtypes.float32, ], _dtypes.float32)
  (input_grads, input_image, output_image) = _inputs_T
  _inputs_flat = [input_grads, input_image, output_image]
  _attrs = ("depth_radius", depth_radius, "bias", bias, "alpha", alpha,
  "beta", beta, "T", _attr_T)
  _result = _execute.execute(b"LRNGrad", 1, inputs=_inputs_flat, attrs=_attrs,
                             ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "LRNGrad", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_LeakyRelu_T = TypeVar("TV_LeakyRelu_T", _atypes.BFloat16, _atypes.Float32, _atypes.Float64, _atypes.Half)

def leaky_relu(features: Annotated[Any, TV_LeakyRelu_T], alpha:float=0.2, name=None) -> Annotated[Any, TV_LeakyRelu_T]:
  r"""Computes rectified linear: `max(features, features * alpha)`.

  Args:
    features: A `Tensor`. Must be one of the following types: `half`, `bfloat16`, `float32`, `float64`.
    alpha: An optional `float`. Defaults to `0.2`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `features`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "LeakyRelu", name, features, "alpha", alpha)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return leaky_relu_eager_fallback(
          features, alpha=alpha, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if alpha is None:
    alpha = 0.2
  alpha = _execute.make_float(alpha, "alpha")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "LeakyRelu", features=features, alpha=alpha, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("alpha", _op.get_attr("alpha"), "T", _op._get_attr_type("T"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "LeakyRelu", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

LeakyRelu = tf_export("raw_ops.LeakyRelu")(_ops.to_raw_op(leaky_relu))


def leaky_relu_eager_fallback(features: Annotated[Any, TV_LeakyRelu_T], alpha: float, name, ctx) -> Annotated[Any, TV_LeakyRelu_T]:
  if alpha is None:
    alpha = 0.2
  alpha = _execute.make_float(alpha, "alpha")
  _attr_T, (features,) = _execute.args_to_matching_eager([features], ctx, [_dtypes.half, _dtypes.bfloat16, _dtypes.float32, _dtypes.float64, ], _dtypes.float32)
  _inputs_flat = [features]
  _attrs = ("alpha", alpha, "T", _attr_T)
  _result = _execute.execute(b"LeakyRelu", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "LeakyRelu", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_LeakyReluGrad_T = TypeVar("TV_LeakyReluGrad_T", _atypes.BFloat16, _atypes.Float32, _atypes.Float64, _atypes.Half)

def leaky_relu_grad(gradients: Annotated[Any, TV_LeakyReluGrad_T], features: Annotated[Any, TV_LeakyReluGrad_T], alpha:float=0.2, name=None) -> Annotated[Any, TV_LeakyReluGrad_T]:
  r"""Computes rectified linear gradients for a LeakyRelu operation.

  Args:
    gradients: A `Tensor`. Must be one of the following types: `half`, `bfloat16`, `float32`, `float64`.
      The backpropagated gradients to the corresponding LeakyRelu operation.
    features: A `Tensor`. Must have the same type as `gradients`.
      The features passed as input to the corresponding LeakyRelu operation,
      OR the outputs of that operation (both work equivalently).
    alpha: An optional `float`. Defaults to `0.2`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `gradients`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "LeakyReluGrad", name, gradients, features, "alpha", alpha)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return leaky_relu_grad_eager_fallback(
          gradients, features, alpha=alpha, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if alpha is None:
    alpha = 0.2
  alpha = _execute.make_float(alpha, "alpha")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "LeakyReluGrad", gradients=gradients, features=features, alpha=alpha,
                         name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("alpha", _op.get_attr("alpha"), "T", _op._get_attr_type("T"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "LeakyReluGrad", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

LeakyReluGrad = tf_export("raw_ops.LeakyReluGrad")(_ops.to_raw_op(leaky_relu_grad))


def leaky_relu_grad_eager_fallback(gradients: Annotated[Any, TV_LeakyReluGrad_T], features: Annotated[Any, TV_LeakyReluGrad_T], alpha: float, name, ctx) -> Annotated[Any, TV_LeakyReluGrad_T]:
  if alpha is None:
    alpha = 0.2
  alpha = _execute.make_float(alpha, "alpha")
  _attr_T, _inputs_T = _execute.args_to_matching_eager([gradients, features], ctx, [_dtypes.half, _dtypes.bfloat16, _dtypes.float32, _dtypes.float64, ], _dtypes.float32)
  (gradients, features) = _inputs_T
  _inputs_flat = [gradients, features]
  _attrs = ("alpha", alpha, "T", _attr_T)
  _result = _execute.execute(b"LeakyReluGrad", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "LeakyReluGrad", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_LogSoftmax_T = TypeVar("TV_LogSoftmax_T", _atypes.BFloat16, _atypes.Float32, _atypes.Float64, _atypes.Half)

def log_softmax(logits: Annotated[Any, TV_LogSoftmax_T], name=None) -> Annotated[Any, TV_LogSoftmax_T]:
  r"""Computes log softmax activations.

  For each batch `i` and class `j` we have

      logsoftmax[i, j] = logits[i, j] - log(sum(exp(logits[i])))

  Args:
    logits: A `Tensor`. Must be one of the following types: `half`, `bfloat16`, `float32`, `float64`.
      2-D with shape `[batch_size, num_classes]`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `logits`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "LogSoftmax", name, logits)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return log_softmax_eager_fallback(
          logits, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "LogSoftmax", logits=logits, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "LogSoftmax", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

LogSoftmax = tf_export("raw_ops.LogSoftmax")(_ops.to_raw_op(log_softmax))


def log_softmax_eager_fallback(logits: Annotated[Any, TV_LogSoftmax_T], name, ctx) -> Annotated[Any, TV_LogSoftmax_T]:
  _attr_T, (logits,) = _execute.args_to_matching_eager([logits], ctx, [_dtypes.half, _dtypes.bfloat16, _dtypes.float32, _dtypes.float64, ])
  _inputs_flat = [logits]
  _attrs = ("T", _attr_T)
  _result = _execute.execute(b"LogSoftmax", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "LogSoftmax", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_MaxPool_T = TypeVar("TV_MaxPool_T", _atypes.BFloat16, _atypes.Float32, _atypes.Float64, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int64, _atypes.Int8, _atypes.QInt8, _atypes.UInt16, _atypes.UInt8)

def max_pool(input: Annotated[Any, TV_MaxPool_T], ksize, strides, padding: str, explicit_paddings=[], data_format:str="NHWC", name=None) -> Annotated[Any, TV_MaxPool_T]:
  r"""Performs max pooling on the input.

  Args:
    input: A `Tensor`. Must be one of the following types: `half`, `bfloat16`, `float32`, `float64`, `int32`, `int64`, `uint8`, `int16`, `int8`, `uint16`, `qint8`.
      4-D input to pool over.
    ksize: A list of `ints` that has length `>= 4`.
      The size of the window for each dimension of the input tensor.
    strides: A list of `ints` that has length `>= 4`.
      The stride of the sliding window for each dimension of the
      input tensor.
    padding: A `string` from: `"SAME", "VALID", "EXPLICIT"`.
      The type of padding algorithm to use.
    explicit_paddings: An optional list of `ints`. Defaults to `[]`.
    data_format: An optional `string` from: `"NHWC", "NCHW", "NCHW_VECT_C"`. Defaults to `"NHWC"`.
      Specify the data format of the input and output data. With the
      default format "NHWC", the data is stored in the order of:
          [batch, in_height, in_width, in_channels].
      Alternatively, the format could be "NCHW", the data storage order of:
          [batch, in_channels, in_height, in_width].
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "MaxPool", name, input, "ksize", ksize, "strides", strides,
        "padding", padding, "explicit_paddings", explicit_paddings,
        "data_format", data_format)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return max_pool_eager_fallback(
          input, ksize=ksize, strides=strides, padding=padding,
          explicit_paddings=explicit_paddings, data_format=data_format,
          name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if not isinstance(ksize, (list, tuple)):
    raise TypeError(
        "Expected list for 'ksize' argument to "
        "'max_pool' Op, not %r." % ksize)
  ksize = [_execute.make_int(_i, "ksize") for _i in ksize]
  if not isinstance(strides, (list, tuple)):
    raise TypeError(
        "Expected list for 'strides' argument to "
        "'max_pool' Op, not %r." % strides)
  strides = [_execute.make_int(_i, "strides") for _i in strides]
  padding = _execute.make_str(padding, "padding")
  if explicit_paddings is None:
    explicit_paddings = []
  if not isinstance(explicit_paddings, (list, tuple)):
    raise TypeError(
        "Expected list for 'explicit_paddings' argument to "
        "'max_pool' Op, not %r." % explicit_paddings)
  explicit_paddings = [_execute.make_int(_i, "explicit_paddings") for _i in explicit_paddings]
  if data_format is None:
    data_format = "NHWC"
  data_format = _execute.make_str(data_format, "data_format")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "MaxPool", input=input, ksize=ksize, strides=strides, padding=padding,
                   explicit_paddings=explicit_paddings,
                   data_format=data_format, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"), "ksize", _op.get_attr("ksize"),
              "strides", _op.get_attr("strides"), "padding",
              _op.get_attr("padding"), "explicit_paddings",
              _op.get_attr("explicit_paddings"), "data_format",
              _op.get_attr("data_format"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "MaxPool", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

MaxPool = tf_export("raw_ops.MaxPool")(_ops.to_raw_op(max_pool))


def max_pool_eager_fallback(input: Annotated[Any, TV_MaxPool_T], ksize, strides, padding: str, explicit_paddings, data_format: str, name, ctx) -> Annotated[Any, TV_MaxPool_T]:
  if not isinstance(ksize, (list, tuple)):
    raise TypeError(
        "Expected list for 'ksize' argument to "
        "'max_pool' Op, not %r." % ksize)
  ksize = [_execute.make_int(_i, "ksize") for _i in ksize]
  if not isinstance(strides, (list, tuple)):
    raise TypeError(
        "Expected list for 'strides' argument to "
        "'max_pool' Op, not %r." % strides)
  strides = [_execute.make_int(_i, "strides") for _i in strides]
  padding = _execute.make_str(padding, "padding")
  if explicit_paddings is None:
    explicit_paddings = []
  if not isinstance(explicit_paddings, (list, tuple)):
    raise TypeError(
        "Expected list for 'explicit_paddings' argument to "
        "'max_pool' Op, not %r." % explicit_paddings)
  explicit_paddings = [_execute.make_int(_i, "explicit_paddings") for _i in explicit_paddings]
  if data_format is None:
    data_format = "NHWC"
  data_format = _execute.make_str(data_format, "data_format")
  _attr_T, (input,) = _execute.args_to_matching_eager([input], ctx, [_dtypes.half, _dtypes.bfloat16, _dtypes.float32, _dtypes.float64, _dtypes.int32, _dtypes.int64, _dtypes.uint8, _dtypes.int16, _dtypes.int8, _dtypes.uint16, _dtypes.qint8, ], _dtypes.float32)
  _inputs_flat = [input]
  _attrs = ("T", _attr_T, "ksize", ksize, "strides", strides, "padding",
  padding, "explicit_paddings", explicit_paddings, "data_format", data_format)
  _result = _execute.execute(b"MaxPool", 1, inputs=_inputs_flat, attrs=_attrs,
                             ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "MaxPool", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_MaxPool3D_T = TypeVar("TV_MaxPool3D_T", _atypes.BFloat16, _atypes.Float32, _atypes.Half)

def max_pool3d(input: Annotated[Any, TV_MaxPool3D_T], ksize, strides, padding: str, data_format:str="NDHWC", name=None) -> Annotated[Any, TV_MaxPool3D_T]:
  r"""Performs 3D max pooling on the input.

  Args:
    input: A `Tensor`. Must be one of the following types: `half`, `bfloat16`, `float32`.
      Shape `[batch, depth, rows, cols, channels]` tensor to pool over.
    ksize: A list of `ints` that has length `>= 5`.
      1-D tensor of length 5. The size of the window for each dimension of
      the input tensor. Must have `ksize[0] = ksize[4] = 1`.
    strides: A list of `ints` that has length `>= 5`.
      1-D tensor of length 5. The stride of the sliding window for each
      dimension of `input`. Must have `strides[0] = strides[4] = 1`.
    padding: A `string` from: `"SAME", "VALID"`.
      The type of padding algorithm to use.
    data_format: An optional `string` from: `"NDHWC", "NCDHW"`. Defaults to `"NDHWC"`.
      The data format of the input and output data. With the
      default format "NDHWC", the data is stored in the order of:
          [batch, in_depth, in_height, in_width, in_channels].
      Alternatively, the format could be "NCDHW", the data storage order is:
          [batch, in_channels, in_depth, in_height, in_width].
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "MaxPool3D", name, input, "ksize", ksize, "strides", strides,
        "padding", padding, "data_format", data_format)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return max_pool3d_eager_fallback(
          input, ksize=ksize, strides=strides, padding=padding,
          data_format=data_format, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if not isinstance(ksize, (list, tuple)):
    raise TypeError(
        "Expected list for 'ksize' argument to "
        "'max_pool3d' Op, not %r." % ksize)
  ksize = [_execute.make_int(_i, "ksize") for _i in ksize]
  if not isinstance(strides, (list, tuple)):
    raise TypeError(
        "Expected list for 'strides' argument to "
        "'max_pool3d' Op, not %r." % strides)
  strides = [_execute.make_int(_i, "strides") for _i in strides]
  padding = _execute.make_str(padding, "padding")
  if data_format is None:
    data_format = "NDHWC"
  data_format = _execute.make_str(data_format, "data_format")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "MaxPool3D", input=input, ksize=ksize, strides=strides,
                     padding=padding, data_format=data_format, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("ksize", _op.get_attr("ksize"), "strides",
              _op.get_attr("strides"), "padding", _op.get_attr("padding"),
              "data_format", _op.get_attr("data_format"), "T",
              _op._get_attr_type("T"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "MaxPool3D", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

MaxPool3D = tf_export("raw_ops.MaxPool3D")(_ops.to_raw_op(max_pool3d))


def max_pool3d_eager_fallback(input: Annotated[Any, TV_MaxPool3D_T], ksize, strides, padding: str, data_format: str, name, ctx) -> Annotated[Any, TV_MaxPool3D_T]:
  if not isinstance(ksize, (list, tuple)):
    raise TypeError(
        "Expected list for 'ksize' argument to "
        "'max_pool3d' Op, not %r." % ksize)
  ksize = [_execute.make_int(_i, "ksize") for _i in ksize]
  if not isinstance(strides, (list, tuple)):
    raise TypeError(
        "Expected list for 'strides' argument to "
        "'max_pool3d' Op, not %r." % strides)
  strides = [_execute.make_int(_i, "strides") for _i in strides]
  padding = _execute.make_str(padding, "padding")
  if data_format is None:
    data_format = "NDHWC"
  data_format = _execute.make_str(data_format, "data_format")
  _attr_T, (input,) = _execute.args_to_matching_eager([input], ctx, [_dtypes.half, _dtypes.bfloat16, _dtypes.float32, ])
  _inputs_flat = [input]
  _attrs = ("ksize", ksize, "strides", strides, "padding", padding,
  "data_format", data_format, "T", _attr_T)
  _result = _execute.execute(b"MaxPool3D", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "MaxPool3D", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_MaxPool3DGrad_T = TypeVar("TV_MaxPool3DGrad_T", _atypes.BFloat16, _atypes.Float32, _atypes.Half)
TV_MaxPool3DGrad_TInput = TypeVar("TV_MaxPool3DGrad_TInput", _atypes.BFloat16, _atypes.Float32, _atypes.Half)

def max_pool3d_grad(orig_input: Annotated[Any, TV_MaxPool3DGrad_TInput], orig_output: Annotated[Any, TV_MaxPool3DGrad_TInput], grad: Annotated[Any, TV_MaxPool3DGrad_T], ksize, strides, padding: str, data_format:str="NDHWC", name=None) -> Annotated[Any, TV_MaxPool3DGrad_T]:
  r"""Computes gradients of 3D max pooling function.

  Args:
    orig_input: A `Tensor`. Must be one of the following types: `half`, `bfloat16`, `float32`.
      The original input tensor.
    orig_output: A `Tensor`. Must have the same type as `orig_input`.
      The original output tensor.
    grad: A `Tensor`. Must be one of the following types: `half`, `bfloat16`, `float32`.
      Output backprop of shape `[batch, depth, rows, cols, channels]`.
    ksize: A list of `ints` that has length `>= 5`.
      1-D tensor of length 5. The size of the window for each dimension of
      the input tensor. Must have `ksize[0] = ksize[4] = 1`.
    strides: A list of `ints` that has length `>= 5`.
      1-D tensor of length 5. The stride of the sliding window for each
      dimension of `input`. Must have `strides[0] = strides[4] = 1`.
    padding: A `string` from: `"SAME", "VALID"`.
      The type of padding algorithm to use.
    data_format: An optional `string` from: `"NDHWC", "NCDHW"`. Defaults to `"NDHWC"`.
      The data format of the input and output data. With the
      default format "NDHWC", the data is stored in the order of:
          [batch, in_depth, in_height, in_width, in_channels].
      Alternatively, the format could be "NCDHW", the data storage order is:
          [batch, in_channels, in_depth, in_height, in_width].
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `grad`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "MaxPool3DGrad", name, orig_input, orig_output, grad, "ksize",
        ksize, "strides", strides, "padding", padding, "data_format",
        data_format)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return max_pool3d_grad_eager_fallback(
          orig_input, orig_output, grad, ksize=ksize, strides=strides,
          padding=padding, data_format=data_format, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if not isinstance(ksize, (list, tuple)):
    raise TypeError(
        "Expected list for 'ksize' argument to "
        "'max_pool3d_grad' Op, not %r." % ksize)
  ksize = [_execute.make_int(_i, "ksize") for _i in ksize]
  if not isinstance(strides, (list, tuple)):
    raise TypeError(
        "Expected list for 'strides' argument to "
        "'max_pool3d_grad' Op, not %r." % strides)
  strides = [_execute.make_int(_i, "strides") for _i in strides]
  padding = _execute.make_str(padding, "padding")
  if data_format is None:
    data_format = "NDHWC"
  data_format = _execute.make_str(data_format, "data_format")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "MaxPool3DGrad", orig_input=orig_input, orig_output=orig_output,
                         grad=grad, ksize=ksize, strides=strides,
                         padding=padding, data_format=data_format, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("ksize", _op.get_attr("ksize"), "strides",
              _op.get_attr("strides"), "padding", _op.get_attr("padding"),
              "data_format", _op.get_attr("data_format"), "T",
              _op._get_attr_type("T"), "TInput", _op._get_attr_type("TInput"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "MaxPool3DGrad", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

MaxPool3DGrad = tf_export("raw_ops.MaxPool3DGrad")(_ops.to_raw_op(max_pool3d_grad))


def max_pool3d_grad_eager_fallback(orig_input: Annotated[Any, TV_MaxPool3DGrad_TInput], orig_output: Annotated[Any, TV_MaxPool3DGrad_TInput], grad: Annotated[Any, TV_MaxPool3DGrad_T], ksize, strides, padding: str, data_format: str, name, ctx) -> Annotated[Any, TV_MaxPool3DGrad_T]:
  if not isinstance(ksize, (list, tuple)):
    raise TypeError(
        "Expected list for 'ksize' argument to "
        "'max_pool3d_grad' Op, not %r." % ksize)
  ksize = [_execute.make_int(_i, "ksize") for _i in ksize]
  if not isinstance(strides, (list, tuple)):
    raise TypeError(
        "Expected list for 'strides' argument to "
        "'max_pool3d_grad' Op, not %r." % strides)
  strides = [_execute.make_int(_i, "strides") for _i in strides]
  padding = _execute.make_str(padding, "padding")
  if data_format is None:
    data_format = "NDHWC"
  data_format = _execute.make_str(data_format, "data_format")
  _attr_T, (grad,) = _execute.args_to_matching_eager([grad], ctx, [_dtypes.half, _dtypes.bfloat16, _dtypes.float32, ], _dtypes.float32)
  _attr_TInput, _inputs_TInput = _execute.args_to_matching_eager([orig_input, orig_output], ctx, [_dtypes.half, _dtypes.bfloat16, _dtypes.float32, ], _dtypes.float32)
  (orig_input, orig_output) = _inputs_TInput
  _inputs_flat = [orig_input, orig_output, grad]
  _attrs = ("ksize", ksize, "strides", strides, "padding", padding,
  "data_format", data_format, "T", _attr_T, "TInput", _attr_TInput)
  _result = _execute.execute(b"MaxPool3DGrad", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "MaxPool3DGrad", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_MaxPool3DGradGrad_T = TypeVar("TV_MaxPool3DGradGrad_T", _atypes.BFloat16, _atypes.Float32, _atypes.Float64, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int64, _atypes.Int8, _atypes.UInt16, _atypes.UInt32, _atypes.UInt64, _atypes.UInt8)

def max_pool3d_grad_grad(orig_input: Annotated[Any, TV_MaxPool3DGradGrad_T], orig_output: Annotated[Any, TV_MaxPool3DGradGrad_T], grad: Annotated[Any, TV_MaxPool3DGradGrad_T], ksize, strides, padding: str, data_format:str="NDHWC", name=None) -> Annotated[Any, TV_MaxPool3DGradGrad_T]:
  r"""Computes second-order gradients of the maxpooling function.

  Args:
    orig_input: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`, `int64`, `bfloat16`, `uint16`, `half`, `uint32`, `uint64`.
      The original input tensor.
    orig_output: A `Tensor`. Must have the same type as `orig_input`.
      The original output tensor.
    grad: A `Tensor`. Must have the same type as `orig_input`.
      Output backprop of shape `[batch, depth, rows, cols, channels]`.
    ksize: A list of `ints` that has length `>= 5`.
      1-D tensor of length 5. The size of the window for each dimension of
      the input tensor. Must have `ksize[0] = ksize[4] = 1`.
    strides: A list of `ints` that has length `>= 5`.
      1-D tensor of length 5. The stride of the sliding window for each
      dimension of `input`. Must have `strides[0] = strides[4] = 1`.
    padding: A `string` from: `"SAME", "VALID"`.
      The type of padding algorithm to use.
    data_format: An optional `string` from: `"NDHWC", "NCDHW"`. Defaults to `"NDHWC"`.
      The data format of the input and output data. With the
      default format "NDHWC", the data is stored in the order of:
          [batch, in_depth, in_height, in_width, in_channels].
      Alternatively, the format could be "NCDHW", the data storage order is:
          [batch, in_channels, in_depth, in_height, in_width].
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `orig_input`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "MaxPool3DGradGrad", name, orig_input, orig_output, grad,
        "ksize", ksize, "strides", strides, "padding", padding, "data_format",
        data_format)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return max_pool3d_grad_grad_eager_fallback(
          orig_input, orig_output, grad, ksize=ksize, strides=strides,
          padding=padding, data_format=data_format, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if not isinstance(ksize, (list, tuple)):
    raise TypeError(
        "Expected list for 'ksize' argument to "
        "'max_pool3d_grad_grad' Op, not %r." % ksize)
  ksize = [_execute.make_int(_i, "ksize") for _i in ksize]
  if not isinstance(strides, (list, tuple)):
    raise TypeError(
        "Expected list for 'strides' argument to "
        "'max_pool3d_grad_grad' Op, not %r." % strides)
  strides = [_execute.make_int(_i, "strides") for _i in strides]
  padding = _execute.make_str(padding, "padding")
  if data_format is None:
    data_format = "NDHWC"
  data_format = _execute.make_str(data_format, "data_format")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "MaxPool3DGradGrad", orig_input=orig_input, orig_output=orig_output,
                             grad=grad, ksize=ksize, strides=strides,
                             padding=padding, data_format=data_format,
                             name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("ksize", _op.get_attr("ksize"), "strides",
              _op.get_attr("strides"), "padding", _op.get_attr("padding"),
              "data_format", _op.get_attr("data_format"), "T",
              _op._get_attr_type("T"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "MaxPool3DGradGrad", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

MaxPool3DGradGrad = tf_export("raw_ops.MaxPool3DGradGrad")(_ops.to_raw_op(max_pool3d_grad_grad))


def max_pool3d_grad_grad_eager_fallback(orig_input: Annotated[Any, TV_MaxPool3DGradGrad_T], orig_output: Annotated[Any, TV_MaxPool3DGradGrad_T], grad: Annotated[Any, TV_MaxPool3DGradGrad_T], ksize, strides, padding: str, data_format: str, name, ctx) -> Annotated[Any, TV_MaxPool3DGradGrad_T]:
  if not isinstance(ksize, (list, tuple)):
    raise TypeError(
        "Expected list for 'ksize' argument to "
        "'max_pool3d_grad_grad' Op, not %r." % ksize)
  ksize = [_execute.make_int(_i, "ksize") for _i in ksize]
  if not isinstance(strides, (list, tuple)):
    raise TypeError(
        "Expected list for 'strides' argument to "
        "'max_pool3d_grad_grad' Op, not %r." % strides)
  strides = [_execute.make_int(_i, "strides") for _i in strides]
  padding = _execute.make_str(padding, "padding")
  if data_format is None:
    data_format = "NDHWC"
  data_format = _execute.make_str(data_format, "data_format")
  _attr_T, _inputs_T = _execute.args_to_matching_eager([orig_input, orig_output, grad], ctx, [_dtypes.float32, _dtypes.float64, _dtypes.int32, _dtypes.uint8, _dtypes.int16, _dtypes.int8, _dtypes.int64, _dtypes.bfloat16, _dtypes.uint16, _dtypes.half, _dtypes.uint32, _dtypes.uint64, ])
  (orig_input, orig_output, grad) = _inputs_T
  _inputs_flat = [orig_input, orig_output, grad]
  _attrs = ("ksize", ksize, "strides", strides, "padding", padding,
  "data_format", data_format, "T", _attr_T)
  _result = _execute.execute(b"MaxPool3DGradGrad", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "MaxPool3DGradGrad", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_MaxPoolGrad_T = TypeVar("TV_MaxPoolGrad_T", _atypes.BFloat16, _atypes.Float32, _atypes.Float64, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int64, _atypes.Int8, _atypes.UInt16, _atypes.UInt32, _atypes.UInt64, _atypes.UInt8)

def max_pool_grad(orig_input: Annotated[Any, TV_MaxPoolGrad_T], orig_output: Annotated[Any, TV_MaxPoolGrad_T], grad: Annotated[Any, TV_MaxPoolGrad_T], ksize, strides, padding: str, explicit_paddings=[], data_format:str="NHWC", name=None) -> Annotated[Any, TV_MaxPoolGrad_T]:
  r"""Computes gradients of the maxpooling function.

  Args:
    orig_input: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`, `int64`, `bfloat16`, `uint16`, `half`, `uint32`, `uint64`.
      The original input tensor.
    orig_output: A `Tensor`. Must have the same type as `orig_input`.
      The original output tensor.
    grad: A `Tensor`. Must have the same type as `orig_input`.
      4-D.  Gradients w.r.t. the output of `max_pool`.
    ksize: A list of `ints` that has length `>= 4`.
      The size of the window for each dimension of the input tensor.
    strides: A list of `ints` that has length `>= 4`.
      The stride of the sliding window for each dimension of the
      input tensor.
    padding: A `string` from: `"SAME", "VALID", "EXPLICIT"`.
      The type of padding algorithm to use.
    explicit_paddings: An optional list of `ints`. Defaults to `[]`.
    data_format: An optional `string` from: `"NHWC", "NCHW"`. Defaults to `"NHWC"`.
      Specify the data format of the input and output data. With the
      default format "NHWC", the data is stored in the order of:
          [batch, in_height, in_width, in_channels].
      Alternatively, the format could be "NCHW", the data storage order of:
          [batch, in_channels, in_height, in_width].
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `orig_input`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "MaxPoolGrad", name, orig_input, orig_output, grad, "ksize",
        ksize, "strides", strides, "padding", padding, "explicit_paddings",
        explicit_paddings, "data_format", data_format)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return max_pool_grad_eager_fallback(
          orig_input, orig_output, grad, ksize=ksize, strides=strides,
          padding=padding, explicit_paddings=explicit_paddings,
          data_format=data_format, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if not isinstance(ksize, (list, tuple)):
    raise TypeError(
        "Expected list for 'ksize' argument to "
        "'max_pool_grad' Op, not %r." % ksize)
  ksize = [_execute.make_int(_i, "ksize") for _i in ksize]
  if not isinstance(strides, (list, tuple)):
    raise TypeError(
        "Expected list for 'strides' argument to "
        "'max_pool_grad' Op, not %r." % strides)
  strides = [_execute.make_int(_i, "strides") for _i in strides]
  padding = _execute.make_str(padding, "padding")
  if explicit_paddings is None:
    explicit_paddings = []
  if not isinstance(explicit_paddings, (list, tuple)):
    raise TypeError(
        "Expected list for 'explicit_paddings' argument to "
        "'max_pool_grad' Op, not %r." % explicit_paddings)
  explicit_paddings = [_execute.make_int(_i, "explicit_paddings") for _i in explicit_paddings]
  if data_format is None:
    data_format = "NHWC"
  data_format = _execute.make_str(data_format, "data_format")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "MaxPoolGrad", orig_input=orig_input, orig_output=orig_output,
                       grad=grad, ksize=ksize, strides=strides,
                       padding=padding, explicit_paddings=explicit_paddings,
                       data_format=data_format, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("ksize", _op.get_attr("ksize"), "strides",
              _op.get_attr("strides"), "padding", _op.get_attr("padding"),
              "explicit_paddings", _op.get_attr("explicit_paddings"),
              "data_format", _op.get_attr("data_format"), "T",
              _op._get_attr_type("T"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "MaxPoolGrad", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

MaxPoolGrad = tf_export("raw_ops.MaxPoolGrad")(_ops.to_raw_op(max_pool_grad))


def max_pool_grad_eager_fallback(orig_input: Annotated[Any, TV_MaxPoolGrad_T], orig_output: Annotated[Any, TV_MaxPoolGrad_T], grad: Annotated[Any, TV_MaxPoolGrad_T], ksize, strides, padding: str, explicit_paddings, data_format: str, name, ctx) -> Annotated[Any, TV_MaxPoolGrad_T]:
  if not isinstance(ksize, (list, tuple)):
    raise TypeError(
        "Expected list for 'ksize' argument to "
        "'max_pool_grad' Op, not %r." % ksize)
  ksize = [_execute.make_int(_i, "ksize") for _i in ksize]
  if not isinstance(strides, (list, tuple)):
    raise TypeError(
        "Expected list for 'strides' argument to "
        "'max_pool_grad' Op, not %r." % strides)
  strides = [_execute.make_int(_i, "strides") for _i in strides]
  padding = _execute.make_str(padding, "padding")
  if explicit_paddings is None:
    explicit_paddings = []
  if not isinstance(explicit_paddings, (list, tuple)):
    raise TypeError(
        "Expected list for 'explicit_paddings' argument to "
        "'max_pool_grad' Op, not %r." % explicit_paddings)
  explicit_paddings = [_execute.make_int(_i, "explicit_paddings") for _i in explicit_paddings]
  if data_format is None:
    data_format = "NHWC"
  data_format = _execute.make_str(data_format, "data_format")
  _attr_T, _inputs_T = _execute.args_to_matching_eager([orig_input, orig_output, grad], ctx, [_dtypes.float32, _dtypes.float64, _dtypes.int32, _dtypes.uint8, _dtypes.int16, _dtypes.int8, _dtypes.int64, _dtypes.bfloat16, _dtypes.uint16, _dtypes.half, _dtypes.uint32, _dtypes.uint64, ], _dtypes.float32)
  (orig_input, orig_output, grad) = _inputs_T
  _inputs_flat = [orig_input, orig_output, grad]
  _attrs = ("ksize", ksize, "strides", strides, "padding", padding,
  "explicit_paddings", explicit_paddings, "data_format", data_format, "T",
  _attr_T)
  _result = _execute.execute(b"MaxPoolGrad", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "MaxPoolGrad", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_MaxPoolGradGrad_T = TypeVar("TV_MaxPoolGradGrad_T", _atypes.BFloat16, _atypes.Float32, _atypes.Float64, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int64, _atypes.Int8, _atypes.UInt16, _atypes.UInt32, _atypes.UInt64, _atypes.UInt8)

def max_pool_grad_grad(orig_input: Annotated[Any, TV_MaxPoolGradGrad_T], orig_output: Annotated[Any, TV_MaxPoolGradGrad_T], grad: Annotated[Any, TV_MaxPoolGradGrad_T], ksize, strides, padding: str, data_format:str="NHWC", name=None) -> Annotated[Any, TV_MaxPoolGradGrad_T]:
  r"""Computes second-order gradients of the maxpooling function.

  Args:
    orig_input: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`, `int64`, `bfloat16`, `uint16`, `half`, `uint32`, `uint64`.
      The original input tensor.
    orig_output: A `Tensor`. Must have the same type as `orig_input`.
      The original output tensor.
    grad: A `Tensor`. Must have the same type as `orig_input`.
      4-D.  Gradients of gradients w.r.t. the input of `max_pool`.
    ksize: A list of `ints` that has length `>= 4`.
      The size of the window for each dimension of the input tensor.
    strides: A list of `ints` that has length `>= 4`.
      The stride of the sliding window for each dimension of the
      input tensor.
    padding: A `string` from: `"SAME", "VALID"`.
      The type of padding algorithm to use.
    data_format: An optional `string` from: `"NHWC", "NCHW"`. Defaults to `"NHWC"`.
      Specify the data format of the input and output data. With the
      default format "NHWC", the data is stored in the order of:
          [batch, in_height, in_width, in_channels].
      Alternatively, the format could be "NCHW", the data storage order of:
          [batch, in_channels, in_height, in_width].
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `orig_input`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "MaxPoolGradGrad", name, orig_input, orig_output, grad, "ksize",
        ksize, "strides", strides, "padding", padding, "data_format",
        data_format)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return max_pool_grad_grad_eager_fallback(
          orig_input, orig_output, grad, ksize=ksize, strides=strides,
          padding=padding, data_format=data_format, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if not isinstance(ksize, (list, tuple)):
    raise TypeError(
        "Expected list for 'ksize' argument to "
        "'max_pool_grad_grad' Op, not %r." % ksize)
  ksize = [_execute.make_int(_i, "ksize") for _i in ksize]
  if not isinstance(strides, (list, tuple)):
    raise TypeError(
        "Expected list for 'strides' argument to "
        "'max_pool_grad_grad' Op, not %r." % strides)
  strides = [_execute.make_int(_i, "strides") for _i in strides]
  padding = _execute.make_str(padding, "padding")
  if data_format is None:
    data_format = "NHWC"
  data_format = _execute.make_str(data_format, "data_format")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "MaxPoolGradGrad", orig_input=orig_input, orig_output=orig_output,
                           grad=grad, ksize=ksize, strides=strides,
                           padding=padding, data_format=data_format,
                           name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("ksize", _op.get_attr("ksize"), "strides",
              _op.get_attr("strides"), "padding", _op.get_attr("padding"),
              "data_format", _op.get_attr("data_format"), "T",
              _op._get_attr_type("T"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "MaxPoolGradGrad", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

MaxPoolGradGrad = tf_export("raw_ops.MaxPoolGradGrad")(_ops.to_raw_op(max_pool_grad_grad))


def max_pool_grad_grad_eager_fallback(orig_input: Annotated[Any, TV_MaxPoolGradGrad_T], orig_output: Annotated[Any, TV_MaxPoolGradGrad_T], grad: Annotated[Any, TV_MaxPoolGradGrad_T], ksize, strides, padding: str, data_format: str, name, ctx) -> Annotated[Any, TV_MaxPoolGradGrad_T]:
  if not isinstance(ksize, (list, tuple)):
    raise TypeError(
        "Expected list for 'ksize' argument to "
        "'max_pool_grad_grad' Op, not %r." % ksize)
  ksize = [_execute.make_int(_i, "ksize") for _i in ksize]
  if not isinstance(strides, (list, tuple)):
    raise TypeError(
        "Expected list for 'strides' argument to "
        "'max_pool_grad_grad' Op, not %r." % strides)
  strides = [_execute.make_int(_i, "strides") for _i in strides]
  padding = _execute.make_str(padding, "padding")
  if data_format is None:
    data_format = "NHWC"
  data_format = _execute.make_str(data_format, "data_format")
  _attr_T, _inputs_T = _execute.args_to_matching_eager([orig_input, orig_output, grad], ctx, [_dtypes.float32, _dtypes.float64, _dtypes.int32, _dtypes.uint8, _dtypes.int16, _dtypes.int8, _dtypes.int64, _dtypes.bfloat16, _dtypes.uint16, _dtypes.half, _dtypes.uint32, _dtypes.uint64, ])
  (orig_input, orig_output, grad) = _inputs_T
  _inputs_flat = [orig_input, orig_output, grad]
  _attrs = ("ksize", ksize, "strides", strides, "padding", padding,
  "data_format", data_format, "T", _attr_T)
  _result = _execute.execute(b"MaxPoolGradGrad", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "MaxPoolGradGrad", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_MaxPoolGradGradV2_T = TypeVar("TV_MaxPoolGradGradV2_T", _atypes.BFloat16, _atypes.Float32, _atypes.Float64, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int64, _atypes.Int8, _atypes.UInt16, _atypes.UInt32, _atypes.UInt64, _atypes.UInt8)

def max_pool_grad_grad_v2(orig_input: Annotated[Any, TV_MaxPoolGradGradV2_T], orig_output: Annotated[Any, TV_MaxPoolGradGradV2_T], grad: Annotated[Any, TV_MaxPoolGradGradV2_T], ksize: Annotated[Any, _atypes.Int32], strides: Annotated[Any, _atypes.Int32], padding: str, data_format:str="NHWC", name=None) -> Annotated[Any, TV_MaxPoolGradGradV2_T]:
  r"""Computes second-order gradients of the maxpooling function.

  Args:
    orig_input: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`, `int64`, `bfloat16`, `uint16`, `half`, `uint32`, `uint64`.
      The original input tensor.
    orig_output: A `Tensor`. Must have the same type as `orig_input`.
      The original output tensor.
    grad: A `Tensor`. Must have the same type as `orig_input`.
      4-D.  Gradients of gradients w.r.t. the input of `max_pool`.
    ksize: A `Tensor` of type `int32`.
      The size of the window for each dimension of the input tensor.
    strides: A `Tensor` of type `int32`.
      The stride of the sliding window for each dimension of the
      input tensor.
    padding: A `string` from: `"SAME", "VALID"`.
      The type of padding algorithm to use.
    data_format: An optional `string` from: `"NHWC", "NCHW"`. Defaults to `"NHWC"`.
      Specify the data format of the input and output data. With the
      default format "NHWC", the data is stored in the order of:
          [batch, in_height, in_width, in_channels].
      Alternatively, the format could be "NCHW", the data storage order of:
          [batch, in_channels, in_height, in_width].
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `orig_input`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "MaxPoolGradGradV2", name, orig_input, orig_output, grad, ksize,
        strides, "padding", padding, "data_format", data_format)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return max_pool_grad_grad_v2_eager_fallback(
          orig_input, orig_output, grad, ksize, strides, padding=padding,
          data_format=data_format, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  padding = _execute.make_str(padding, "padding")
  if data_format is None:
    data_format = "NHWC"
  data_format = _execute.make_str(data_format, "data_format")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "MaxPoolGradGradV2", orig_input=orig_input, orig_output=orig_output,
                             grad=grad, ksize=ksize, strides=strides,
                             padding=padding, data_format=data_format,
                             name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("padding", _op.get_attr("padding"), "data_format",
              _op.get_attr("data_format"), "T", _op._get_attr_type("T"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "MaxPoolGradGradV2", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

MaxPoolGradGradV2 = tf_export("raw_ops.MaxPoolGradGradV2")(_ops.to_raw_op(max_pool_grad_grad_v2))


def max_pool_grad_grad_v2_eager_fallback(orig_input: Annotated[Any, TV_MaxPoolGradGradV2_T], orig_output: Annotated[Any, TV_MaxPoolGradGradV2_T], grad: Annotated[Any, TV_MaxPoolGradGradV2_T], ksize: Annotated[Any, _atypes.Int32], strides: Annotated[Any, _atypes.Int32], padding: str, data_format: str, name, ctx) -> Annotated[Any, TV_MaxPoolGradGradV2_T]:
  padding = _execute.make_str(padding, "padding")
  if data_format is None:
    data_format = "NHWC"
  data_format = _execute.make_str(data_format, "data_format")
  _attr_T, _inputs_T = _execute.args_to_matching_eager([orig_input, orig_output, grad], ctx, [_dtypes.float32, _dtypes.float64, _dtypes.int32, _dtypes.uint8, _dtypes.int16, _dtypes.int8, _dtypes.int64, _dtypes.bfloat16, _dtypes.uint16, _dtypes.half, _dtypes.uint32, _dtypes.uint64, ])
  (orig_input, orig_output, grad) = _inputs_T
  ksize = _ops.convert_to_tensor(ksize, _dtypes.int32)
  strides = _ops.convert_to_tensor(strides, _dtypes.int32)
  _inputs_flat = [orig_input, orig_output, grad, ksize, strides]
  _attrs = ("padding", padding, "data_format", data_format, "T", _attr_T)
  _result = _execute.execute(b"MaxPoolGradGradV2", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "MaxPoolGradGradV2", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_MaxPoolGradGradWithArgmax_Targmax = TypeVar("TV_MaxPoolGradGradWithArgmax_Targmax", _atypes.Int32, _atypes.Int64)
TV_MaxPoolGradGradWithArgmax_T = TypeVar("TV_MaxPoolGradGradWithArgmax_T", _atypes.BFloat16, _atypes.Float32, _atypes.Float64, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int64, _atypes.Int8, _atypes.UInt16, _atypes.UInt32, _atypes.UInt64, _atypes.UInt8)

def max_pool_grad_grad_with_argmax(input: Annotated[Any, TV_MaxPoolGradGradWithArgmax_T], grad: Annotated[Any, TV_MaxPoolGradGradWithArgmax_T], argmax: Annotated[Any, TV_MaxPoolGradGradWithArgmax_Targmax], ksize, strides, padding: str, include_batch_in_index:bool=False, name=None) -> Annotated[Any, TV_MaxPoolGradGradWithArgmax_T]:
  r"""Computes second-order gradients of the maxpooling function.

  Args:
    input: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`, `int64`, `bfloat16`, `uint16`, `half`, `uint32`, `uint64`.
      The original input.
    grad: A `Tensor`. Must have the same type as `input`.
      4-D with shape `[batch, height, width, channels]`.  Gradients w.r.t. the
      input of `max_pool`.
    argmax: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      The indices of the maximum values chosen for each output of `max_pool`.
    ksize: A list of `ints` that has length `>= 4`.
      The size of the window for each dimension of the input tensor.
    strides: A list of `ints` that has length `>= 4`.
      The stride of the sliding window for each dimension of the
      input tensor.
    padding: A `string` from: `"SAME", "VALID"`.
      The type of padding algorithm to use.
    include_batch_in_index: An optional `bool`. Defaults to `False`.
      Whether to include batch dimension in flattened index of `argmax`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "MaxPoolGradGradWithArgmax", name, input, grad, argmax, "ksize",
        ksize, "strides", strides, "padding", padding,
        "include_batch_in_index", include_batch_in_index)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return max_pool_grad_grad_with_argmax_eager_fallback(
          input, grad, argmax, ksize=ksize, strides=strides, padding=padding,
          include_batch_in_index=include_batch_in_index, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if not isinstance(ksize, (list, tuple)):
    raise TypeError(
        "Expected list for 'ksize' argument to "
        "'max_pool_grad_grad_with_argmax' Op, not %r." % ksize)
  ksize = [_execute.make_int(_i, "ksize") for _i in ksize]
  if not isinstance(strides, (list, tuple)):
    raise TypeError(
        "Expected list for 'strides' argument to "
        "'max_pool_grad_grad_with_argmax' Op, not %r." % strides)
  strides = [_execute.make_int(_i, "strides") for _i in strides]
  padding = _execute.make_str(padding, "padding")
  if include_batch_in_index is None:
    include_batch_in_index = False
  include_batch_in_index = _execute.make_bool(include_batch_in_index, "include_batch_in_index")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "MaxPoolGradGradWithArgmax", input=input, grad=grad, argmax=argmax,
                                     ksize=ksize, strides=strides,
                                     padding=padding,
                                     include_batch_in_index=include_batch_in_index,
                                     name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("ksize", _op.get_attr("ksize"), "strides",
              _op.get_attr("strides"), "padding", _op.get_attr("padding"),
              "include_batch_in_index",
              _op._get_attr_bool("include_batch_in_index"), "Targmax",
              _op._get_attr_type("Targmax"), "T", _op._get_attr_type("T"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "MaxPoolGradGradWithArgmax", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

MaxPoolGradGradWithArgmax = tf_export("raw_ops.MaxPoolGradGradWithArgmax")(_ops.to_raw_op(max_pool_grad_grad_with_argmax))


def max_pool_grad_grad_with_argmax_eager_fallback(input: Annotated[Any, TV_MaxPoolGradGradWithArgmax_T], grad: Annotated[Any, TV_MaxPoolGradGradWithArgmax_T], argmax: Annotated[Any, TV_MaxPoolGradGradWithArgmax_Targmax], ksize, strides, padding: str, include_batch_in_index: bool, name, ctx) -> Annotated[Any, TV_MaxPoolGradGradWithArgmax_T]:
  if not isinstance(ksize, (list, tuple)):
    raise TypeError(
        "Expected list for 'ksize' argument to "
        "'max_pool_grad_grad_with_argmax' Op, not %r." % ksize)
  ksize = [_execute.make_int(_i, "ksize") for _i in ksize]
  if not isinstance(strides, (list, tuple)):
    raise TypeError(
        "Expected list for 'strides' argument to "
        "'max_pool_grad_grad_with_argmax' Op, not %r." % strides)
  strides = [_execute.make_int(_i, "strides") for _i in strides]
  padding = _execute.make_str(padding, "padding")
  if include_batch_in_index is None:
    include_batch_in_index = False
  include_batch_in_index = _execute.make_bool(include_batch_in_index, "include_batch_in_index")
  _attr_Targmax, (argmax,) = _execute.args_to_matching_eager([argmax], ctx, [_dtypes.int32, _dtypes.int64, ])
  _attr_T, _inputs_T = _execute.args_to_matching_eager([input, grad], ctx, [_dtypes.float32, _dtypes.float64, _dtypes.int32, _dtypes.uint8, _dtypes.int16, _dtypes.int8, _dtypes.int64, _dtypes.bfloat16, _dtypes.uint16, _dtypes.half, _dtypes.uint32, _dtypes.uint64, ])
  (input, grad) = _inputs_T
  _inputs_flat = [input, grad, argmax]
  _attrs = ("ksize", ksize, "strides", strides, "padding", padding,
  "include_batch_in_index", include_batch_in_index, "Targmax", _attr_Targmax,
  "T", _attr_T)
  _result = _execute.execute(b"MaxPoolGradGradWithArgmax", 1,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "MaxPoolGradGradWithArgmax", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_MaxPoolGradV2_T = TypeVar("TV_MaxPoolGradV2_T", _atypes.BFloat16, _atypes.Float32, _atypes.Float64, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int64, _atypes.Int8, _atypes.UInt16, _atypes.UInt32, _atypes.UInt64, _atypes.UInt8)

def max_pool_grad_v2(orig_input: Annotated[Any, TV_MaxPoolGradV2_T], orig_output: Annotated[Any, TV_MaxPoolGradV2_T], grad: Annotated[Any, TV_MaxPoolGradV2_T], ksize: Annotated[Any, _atypes.Int32], strides: Annotated[Any, _atypes.Int32], padding: str, data_format:str="NHWC", name=None) -> Annotated[Any, TV_MaxPoolGradV2_T]:
  r"""Computes gradients of the maxpooling function.

  Args:
    orig_input: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`, `int64`, `bfloat16`, `uint16`, `half`, `uint32`, `uint64`.
      The original input tensor.
    orig_output: A `Tensor`. Must have the same type as `orig_input`.
      The original output tensor.
    grad: A `Tensor`. Must have the same type as `orig_input`.
      4-D.  Gradients w.r.t. the output of `max_pool`.
    ksize: A `Tensor` of type `int32`.
      The size of the window for each dimension of the input tensor.
    strides: A `Tensor` of type `int32`.
      The stride of the sliding window for each dimension of the
      input tensor.
    padding: A `string` from: `"SAME", "VALID"`.
      The type of padding algorithm to use.
    data_format: An optional `string` from: `"NHWC", "NCHW"`. Defaults to `"NHWC"`.
      Specify the data format of the input and output data. With the
      default format "NHWC", the data is stored in the order of:
          [batch, in_height, in_width, in_channels].
      Alternatively, the format could be "NCHW", the data storage order of:
          [batch, in_channels, in_height, in_width].
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `orig_input`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "MaxPoolGradV2", name, orig_input, orig_output, grad, ksize,
        strides, "padding", padding, "data_format", data_format)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return max_pool_grad_v2_eager_fallback(
          orig_input, orig_output, grad, ksize, strides, padding=padding,
          data_format=data_format, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  padding = _execute.make_str(padding, "padding")
  if data_format is None:
    data_format = "NHWC"
  data_format = _execute.make_str(data_format, "data_format")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "MaxPoolGradV2", orig_input=orig_input, orig_output=orig_output,
                         grad=grad, ksize=ksize, strides=strides,
                         padding=padding, data_format=data_format, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("padding", _op.get_attr("padding"), "data_format",
              _op.get_attr("data_format"), "T", _op._get_attr_type("T"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "MaxPoolGradV2", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

MaxPoolGradV2 = tf_export("raw_ops.MaxPoolGradV2")(_ops.to_raw_op(max_pool_grad_v2))


def max_pool_grad_v2_eager_fallback(orig_input: Annotated[Any, TV_MaxPoolGradV2_T], orig_output: Annotated[Any, TV_MaxPoolGradV2_T], grad: Annotated[Any, TV_MaxPoolGradV2_T], ksize: Annotated[Any, _atypes.Int32], strides: Annotated[Any, _atypes.Int32], padding: str, data_format: str, name, ctx) -> Annotated[Any, TV_MaxPoolGradV2_T]:
  padding = _execute.make_str(padding, "padding")
  if data_format is None:
    data_format = "NHWC"
  data_format = _execute.make_str(data_format, "data_format")
  _attr_T, _inputs_T = _execute.args_to_matching_eager([orig_input, orig_output, grad], ctx, [_dtypes.float32, _dtypes.float64, _dtypes.int32, _dtypes.uint8, _dtypes.int16, _dtypes.int8, _dtypes.int64, _dtypes.bfloat16, _dtypes.uint16, _dtypes.half, _dtypes.uint32, _dtypes.uint64, ], _dtypes.float32)
  (orig_input, orig_output, grad) = _inputs_T
  ksize = _ops.convert_to_tensor(ksize, _dtypes.int32)
  strides = _ops.convert_to_tensor(strides, _dtypes.int32)
  _inputs_flat = [orig_input, orig_output, grad, ksize, strides]
  _attrs = ("padding", padding, "data_format", data_format, "T", _attr_T)
  _result = _execute.execute(b"MaxPoolGradV2", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "MaxPoolGradV2", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_MaxPoolGradWithArgmax_Targmax = TypeVar("TV_MaxPoolGradWithArgmax_Targmax", _atypes.Int32, _atypes.Int64)
TV_MaxPoolGradWithArgmax_T = TypeVar("TV_MaxPoolGradWithArgmax_T", _atypes.BFloat16, _atypes.Float32, _atypes.Float64, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int64, _atypes.Int8, _atypes.UInt16, _atypes.UInt32, _atypes.UInt64, _atypes.UInt8)

def max_pool_grad_with_argmax(input: Annotated[Any, TV_MaxPoolGradWithArgmax_T], grad: Annotated[Any, TV_MaxPoolGradWithArgmax_T], argmax: Annotated[Any, TV_MaxPoolGradWithArgmax_Targmax], ksize, strides, padding: str, include_batch_in_index:bool=False, name=None) -> Annotated[Any, TV_MaxPoolGradWithArgmax_T]:
  r"""Computes gradients of the maxpooling function.

  Args:
    input: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`, `int64`, `bfloat16`, `uint16`, `half`, `uint32`, `uint64`.
      The original input.
    grad: A `Tensor`. Must have the same type as `input`.
      4-D with shape `[batch, height, width, channels]`.  Gradients w.r.t. the
      output of `max_pool`.
    argmax: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      The indices of the maximum values chosen for each output of `max_pool`.
    ksize: A list of `ints` that has length `>= 4`.
      The size of the window for each dimension of the input tensor.
    strides: A list of `ints` that has length `>= 4`.
      The stride of the sliding window for each dimension of the
      input tensor.
    padding: A `string` from: `"SAME", "VALID"`.
      The type of padding algorithm to use.
    include_batch_in_index: An optional `bool`. Defaults to `False`.
      Whether to include batch dimension in flattened index of `argmax`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "MaxPoolGradWithArgmax", name, input, grad, argmax, "ksize",
        ksize, "strides", strides, "padding", padding,
        "include_batch_in_index", include_batch_in_index)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return max_pool_grad_with_argmax_eager_fallback(
          input, grad, argmax, ksize=ksize, strides=strides, padding=padding,
          include_batch_in_index=include_batch_in_index, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if not isinstance(ksize, (list, tuple)):
    raise TypeError(
        "Expected list for 'ksize' argument to "
        "'max_pool_grad_with_argmax' Op, not %r." % ksize)
  ksize = [_execute.make_int(_i, "ksize") for _i in ksize]
  if not isinstance(strides, (list, tuple)):
    raise TypeError(
        "Expected list for 'strides' argument to "
        "'max_pool_grad_with_argmax' Op, not %r." % strides)
  strides = [_execute.make_int(_i, "strides") for _i in strides]
  padding = _execute.make_str(padding, "padding")
  if include_batch_in_index is None:
    include_batch_in_index = False
  include_batch_in_index = _execute.make_bool(include_batch_in_index, "include_batch_in_index")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "MaxPoolGradWithArgmax", input=input, grad=grad, argmax=argmax,
                                 ksize=ksize, strides=strides,
                                 padding=padding,
                                 include_batch_in_index=include_batch_in_index,
                                 name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("ksize", _op.get_attr("ksize"), "strides",
              _op.get_attr("strides"), "padding", _op.get_attr("padding"),
              "include_batch_in_index",
              _op._get_attr_bool("include_batch_in_index"), "Targmax",
              _op._get_attr_type("Targmax"), "T", _op._get_attr_type("T"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "MaxPoolGradWithArgmax", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

MaxPoolGradWithArgmax = tf_export("raw_ops.MaxPoolGradWithArgmax")(_ops.to_raw_op(max_pool_grad_with_argmax))


def max_pool_grad_with_argmax_eager_fallback(input: Annotated[Any, TV_MaxPoolGradWithArgmax_T], grad: Annotated[Any, TV_MaxPoolGradWithArgmax_T], argmax: Annotated[Any, TV_MaxPoolGradWithArgmax_Targmax], ksize, strides, padding: str, include_batch_in_index: bool, name, ctx) -> Annotated[Any, TV_MaxPoolGradWithArgmax_T]:
  if not isinstance(ksize, (list, tuple)):
    raise TypeError(
        "Expected list for 'ksize' argument to "
        "'max_pool_grad_with_argmax' Op, not %r." % ksize)
  ksize = [_execute.make_int(_i, "ksize") for _i in ksize]
  if not isinstance(strides, (list, tuple)):
    raise TypeError(
        "Expected list for 'strides' argument to "
        "'max_pool_grad_with_argmax' Op, not %r." % strides)
  strides = [_execute.make_int(_i, "strides") for _i in strides]
  padding = _execute.make_str(padding, "padding")
  if include_batch_in_index is None:
    include_batch_in_index = False
  include_batch_in_index = _execute.make_bool(include_batch_in_index, "include_batch_in_index")
  _attr_Targmax, (argmax,) = _execute.args_to_matching_eager([argmax], ctx, [_dtypes.int32, _dtypes.int64, ])
  _attr_T, _inputs_T = _execute.args_to_matching_eager([input, grad], ctx, [_dtypes.float32, _dtypes.float64, _dtypes.int32, _dtypes.uint8, _dtypes.int16, _dtypes.int8, _dtypes.int64, _dtypes.bfloat16, _dtypes.uint16, _dtypes.half, _dtypes.uint32, _dtypes.uint64, ])
  (input, grad) = _inputs_T
  _inputs_flat = [input, grad, argmax]
  _attrs = ("ksize", ksize, "strides", strides, "padding", padding,
  "include_batch_in_index", include_batch_in_index, "Targmax", _attr_Targmax,
  "T", _attr_T)
  _result = _execute.execute(b"MaxPoolGradWithArgmax", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "MaxPoolGradWithArgmax", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_MaxPoolV2_T = TypeVar("TV_MaxPoolV2_T", _atypes.BFloat16, _atypes.Float32, _atypes.Float64, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int64, _atypes.Int8, _atypes.QInt8, _atypes.UInt16, _atypes.UInt8)

def max_pool_v2(input: Annotated[Any, TV_MaxPoolV2_T], ksize: Annotated[Any, _atypes.Int32], strides: Annotated[Any, _atypes.Int32], padding: str, data_format:str="NHWC", name=None) -> Annotated[Any, TV_MaxPoolV2_T]:
  r"""Performs max pooling on the input.

  Args:
    input: A `Tensor`. Must be one of the following types: `half`, `bfloat16`, `float32`, `float64`, `int32`, `int64`, `uint8`, `int16`, `int8`, `uint16`, `qint8`.
      4-D input to pool over.
    ksize: A `Tensor` of type `int32`.
      The size of the window for each dimension of the input tensor.
    strides: A `Tensor` of type `int32`.
      The stride of the sliding window for each dimension of the
      input tensor.
    padding: A `string` from: `"SAME", "VALID"`.
      The type of padding algorithm to use.
    data_format: An optional `string` from: `"NHWC", "NCHW", "NCHW_VECT_C"`. Defaults to `"NHWC"`.
      Specify the data format of the input and output data. With the
      default format "NHWC", the data is stored in the order of:
          [batch, in_height, in_width, in_channels].
      Alternatively, the format could be "NCHW", the data storage order of:
          [batch, in_channels, in_height, in_width].
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "MaxPoolV2", name, input, ksize, strides, "padding", padding,
        "data_format", data_format)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return max_pool_v2_eager_fallback(
          input, ksize, strides, padding=padding, data_format=data_format,
          name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  padding = _execute.make_str(padding, "padding")
  if data_format is None:
    data_format = "NHWC"
  data_format = _execute.make_str(data_format, "data_format")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "MaxPoolV2", input=input, ksize=ksize, strides=strides,
                     padding=padding, data_format=data_format, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"), "padding",
              _op.get_attr("padding"), "data_format",
              _op.get_attr("data_format"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "MaxPoolV2", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

MaxPoolV2 = tf_export("raw_ops.MaxPoolV2")(_ops.to_raw_op(max_pool_v2))


def max_pool_v2_eager_fallback(input: Annotated[Any, TV_MaxPoolV2_T], ksize: Annotated[Any, _atypes.Int32], strides: Annotated[Any, _atypes.Int32], padding: str, data_format: str, name, ctx) -> Annotated[Any, TV_MaxPoolV2_T]:
  padding = _execute.make_str(padding, "padding")
  if data_format is None:
    data_format = "NHWC"
  data_format = _execute.make_str(data_format, "data_format")
  _attr_T, (input,) = _execute.args_to_matching_eager([input], ctx, [_dtypes.half, _dtypes.bfloat16, _dtypes.float32, _dtypes.float64, _dtypes.int32, _dtypes.int64, _dtypes.uint8, _dtypes.int16, _dtypes.int8, _dtypes.uint16, _dtypes.qint8, ], _dtypes.float32)
  ksize = _ops.convert_to_tensor(ksize, _dtypes.int32)
  strides = _ops.convert_to_tensor(strides, _dtypes.int32)
  _inputs_flat = [input, ksize, strides]
  _attrs = ("T", _attr_T, "padding", padding, "data_format", data_format)
  _result = _execute.execute(b"MaxPoolV2", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "MaxPoolV2", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

_MaxPoolWithArgmaxOutput = collections.namedtuple(
    "MaxPoolWithArgmax",
    ["output", "argmax"])


TV_MaxPoolWithArgmax_Targmax = TypeVar("TV_MaxPoolWithArgmax_Targmax", _atypes.Int32, _atypes.Int64)
TV_MaxPoolWithArgmax_T = TypeVar("TV_MaxPoolWithArgmax_T", _atypes.BFloat16, _atypes.Float32, _atypes.Float64, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int64, _atypes.Int8, _atypes.UInt16, _atypes.UInt32, _atypes.UInt64, _atypes.UInt8)

def max_pool_with_argmax(input: Annotated[Any, TV_MaxPoolWithArgmax_T], ksize, strides, padding: str, Targmax:TV_MaxPoolWithArgmax_Targmax=_dtypes.int64, include_batch_in_index:bool=False, name=None):
  r"""Performs max pooling on the input and outputs both max values and indices.

  The indices in `argmax` are flattened, so that a maximum value at position
  `[b, y, x, c]` becomes flattened index:
  `(y * width + x) * channels + c` if `include_batch_in_index` is False;
  `((b * height + y) * width + x) * channels + c` if `include_batch_in_index` is True.

  The indices returned are always in `[0, height) x [0, width)` before flattening,
  even if padding is involved and the mathematically correct answer is outside
  (either negative or too large).  This is a bug, but fixing it is difficult to do
  in a safe backwards compatible way, especially due to flattening.

  Args:
    input: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`, `int64`, `bfloat16`, `uint16`, `half`, `uint32`, `uint64`.
      4-D with shape `[batch, height, width, channels]`.  Input to pool over.
    ksize: A list of `ints` that has length `>= 4`.
      The size of the window for each dimension of the input tensor.
    strides: A list of `ints` that has length `>= 4`.
      The stride of the sliding window for each dimension of the
      input tensor.
    padding: A `string` from: `"SAME", "VALID"`.
      The type of padding algorithm to use.
    Targmax: An optional `tf.DType` from: `tf.int32, tf.int64`. Defaults to `tf.int64`.
    include_batch_in_index: An optional `bool`. Defaults to `False`.
      Whether to include batch dimension in flattened index of `argmax`.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (output, argmax).

    output: A `Tensor`. Has the same type as `input`.
    argmax: A `Tensor` of type `Targmax`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "MaxPoolWithArgmax", name, input, "ksize", ksize, "strides",
        strides, "Targmax", Targmax, "padding", padding,
        "include_batch_in_index", include_batch_in_index)
      _result = _MaxPoolWithArgmaxOutput._make(_result)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return max_pool_with_argmax_eager_fallback(
          input, ksize=ksize, strides=strides, Targmax=Targmax,
          padding=padding, include_batch_in_index=include_batch_in_index,
          name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if not isinstance(ksize, (list, tuple)):
    raise TypeError(
        "Expected list for 'ksize' argument to "
        "'max_pool_with_argmax' Op, not %r." % ksize)
  ksize = [_execute.make_int(_i, "ksize") for _i in ksize]
  if not isinstance(strides, (list, tuple)):
    raise TypeError(
        "Expected list for 'strides' argument to "
        "'max_pool_with_argmax' Op, not %r." % strides)
  strides = [_execute.make_int(_i, "strides") for _i in strides]
  padding = _execute.make_str(padding, "padding")
  if Targmax is None:
    Targmax = _dtypes.int64
  Targmax = _execute.make_type(Targmax, "Targmax")
  if include_batch_in_index is None:
    include_batch_in_index = False
  include_batch_in_index = _execute.make_bool(include_batch_in_index, "include_batch_in_index")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "MaxPoolWithArgmax", input=input, ksize=ksize, strides=strides,
                             padding=padding, Targmax=Targmax,
                             include_batch_in_index=include_batch_in_index,
                             name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("ksize", _op.get_attr("ksize"), "strides",
              _op.get_attr("strides"), "Targmax",
              _op._get_attr_type("Targmax"), "padding",
              _op.get_attr("padding"), "include_batch_in_index",
              _op._get_attr_bool("include_batch_in_index"), "T",
              _op._get_attr_type("T"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "MaxPoolWithArgmax", _inputs_flat, _attrs, _result)
  _result = _MaxPoolWithArgmaxOutput._make(_result)
  return _result

MaxPoolWithArgmax = tf_export("raw_ops.MaxPoolWithArgmax")(_ops.to_raw_op(max_pool_with_argmax))


def max_pool_with_argmax_eager_fallback(input: Annotated[Any, TV_MaxPoolWithArgmax_T], ksize, strides, padding: str, Targmax: TV_MaxPoolWithArgmax_Targmax, include_batch_in_index: bool, name, ctx):
  if not isinstance(ksize, (list, tuple)):
    raise TypeError(
        "Expected list for 'ksize' argument to "
        "'max_pool_with_argmax' Op, not %r." % ksize)
  ksize = [_execute.make_int(_i, "ksize") for _i in ksize]
  if not isinstance(strides, (list, tuple)):
    raise TypeError(
        "Expected list for 'strides' argument to "
        "'max_pool_with_argmax' Op, not %r." % strides)
  strides = [_execute.make_int(_i, "strides") for _i in strides]
  padding = _execute.make_str(padding, "padding")
  if Targmax is None:
    Targmax = _dtypes.int64
  Targmax = _execute.make_type(Targmax, "Targmax")
  if include_batch_in_index is None:
    include_batch_in_index = False
  include_batch_in_index = _execute.make_bool(include_batch_in_index, "include_batch_in_index")
  _attr_T, (input,) = _execute.args_to_matching_eager([input], ctx, [_dtypes.float32, _dtypes.float64, _dtypes.int32, _dtypes.uint8, _dtypes.int16, _dtypes.int8, _dtypes.int64, _dtypes.bfloat16, _dtypes.uint16, _dtypes.half, _dtypes.uint32, _dtypes.uint64, ])
  _inputs_flat = [input]
  _attrs = ("ksize", ksize, "strides", strides, "Targmax", Targmax, "padding",
  padding, "include_batch_in_index", include_batch_in_index, "T", _attr_T)
  _result = _execute.execute(b"MaxPoolWithArgmax", 2, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "MaxPoolWithArgmax", _inputs_flat, _attrs, _result)
  _result = _MaxPoolWithArgmaxOutput._make(_result)
  return _result


TV_NthElement_T = TypeVar("TV_NthElement_T", _atypes.BFloat16, _atypes.Float32, _atypes.Float64, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int64, _atypes.Int8, _atypes.UInt16, _atypes.UInt32, _atypes.UInt64, _atypes.UInt8)

def nth_element(input: Annotated[Any, TV_NthElement_T], n: Annotated[Any, _atypes.Int32], reverse:bool=False, name=None) -> Annotated[Any, TV_NthElement_T]:
  r"""Finds values of the `n`-th order statistic for the last dimension.

  If the input is a vector (rank-1), finds the entries which is the nth-smallest
  value in the vector and outputs their values as scalar tensor.

  For matrices (resp. higher rank input), computes the entries which is the
  nth-smallest value in each row (resp. vector along the last dimension). Thus,

      values.shape = input.shape[:-1]

  Args:
    input: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`, `int64`, `bfloat16`, `uint16`, `half`, `uint32`, `uint64`.
      1-D or higher with last dimension at least `n+1`.
    n: A `Tensor` of type `int32`.
      0-D. Position of sorted vector to select along the last dimension (along
      each row for matrices). Valid range of n is `[0, input.shape[:-1])`
    reverse: An optional `bool`. Defaults to `False`.
      When set to True, find the nth-largest value in the vector and vice
      versa.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "NthElement", name, input, n, "reverse", reverse)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return nth_element_eager_fallback(
          input, n, reverse=reverse, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if reverse is None:
    reverse = False
  reverse = _execute.make_bool(reverse, "reverse")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "NthElement", input=input, n=n, reverse=reverse, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("reverse", _op._get_attr_bool("reverse"), "T",
              _op._get_attr_type("T"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "NthElement", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

NthElement = tf_export("raw_ops.NthElement")(_ops.to_raw_op(nth_element))


def nth_element_eager_fallback(input: Annotated[Any, TV_NthElement_T], n: Annotated[Any, _atypes.Int32], reverse: bool, name, ctx) -> Annotated[Any, TV_NthElement_T]:
  if reverse is None:
    reverse = False
  reverse = _execute.make_bool(reverse, "reverse")
  _attr_T, (input,) = _execute.args_to_matching_eager([input], ctx, [_dtypes.float32, _dtypes.float64, _dtypes.int32, _dtypes.uint8, _dtypes.int16, _dtypes.int8, _dtypes.int64, _dtypes.bfloat16, _dtypes.uint16, _dtypes.half, _dtypes.uint32, _dtypes.uint64, ])
  n = _ops.convert_to_tensor(n, _dtypes.int32)
  _inputs_flat = [input, n]
  _attrs = ("reverse", reverse, "T", _attr_T)
  _result = _execute.execute(b"NthElement", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "NthElement", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

_QuantizedAvgPoolOutput = collections.namedtuple(
    "QuantizedAvgPool",
    ["output", "min_output", "max_output"])


TV_QuantizedAvgPool_T = TypeVar("TV_QuantizedAvgPool_T", _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8)

def quantized_avg_pool(input: Annotated[Any, TV_QuantizedAvgPool_T], min_input: Annotated[Any, _atypes.Float32], max_input: Annotated[Any, _atypes.Float32], ksize, strides, padding: str, name=None):
  r"""Produces the average pool of the input tensor for quantized types.

  Args:
    input: A `Tensor`. Must be one of the following types: `qint8`, `quint8`, `qint32`, `qint16`, `quint16`.
      4-D with shape `[batch, height, width, channels]`.
    min_input: A `Tensor` of type `float32`.
      The float value that the lowest quantized input value represents.
    max_input: A `Tensor` of type `float32`.
      The float value that the highest quantized input value represents.
    ksize: A list of `ints`.
      The size of the window for each dimension of the input tensor.
      The length must be 4 to match the number of dimensions of the input.
    strides: A list of `ints`.
      The stride of the sliding window for each dimension of the input
      tensor.  The length must be 4 to match the number of dimensions of the input.
    padding: A `string` from: `"SAME", "VALID"`.
      The type of padding algorithm to use.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (output, min_output, max_output).

    output: A `Tensor`. Has the same type as `input`.
    min_output: A `Tensor` of type `float32`.
    max_output: A `Tensor` of type `float32`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "QuantizedAvgPool", name, input, min_input, max_input, "ksize",
        ksize, "strides", strides, "padding", padding)
      _result = _QuantizedAvgPoolOutput._make(_result)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return quantized_avg_pool_eager_fallback(
          input, min_input, max_input, ksize=ksize, strides=strides,
          padding=padding, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if not isinstance(ksize, (list, tuple)):
    raise TypeError(
        "Expected list for 'ksize' argument to "
        "'quantized_avg_pool' Op, not %r." % ksize)
  ksize = [_execute.make_int(_i, "ksize") for _i in ksize]
  if not isinstance(strides, (list, tuple)):
    raise TypeError(
        "Expected list for 'strides' argument to "
        "'quantized_avg_pool' Op, not %r." % strides)
  strides = [_execute.make_int(_i, "strides") for _i in strides]
  padding = _execute.make_str(padding, "padding")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "QuantizedAvgPool", input=input, min_input=min_input,
                            max_input=max_input, ksize=ksize, strides=strides,
                            padding=padding, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"), "ksize", _op.get_attr("ksize"),
              "strides", _op.get_attr("strides"), "padding",
              _op.get_attr("padding"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "QuantizedAvgPool", _inputs_flat, _attrs, _result)
  _result = _QuantizedAvgPoolOutput._make(_result)
  return _result

QuantizedAvgPool = tf_export("raw_ops.QuantizedAvgPool")(_ops.to_raw_op(quantized_avg_pool))


def quantized_avg_pool_eager_fallback(input: Annotated[Any, TV_QuantizedAvgPool_T], min_input: Annotated[Any, _atypes.Float32], max_input: Annotated[Any, _atypes.Float32], ksize, strides, padding: str, name, ctx):
  if not isinstance(ksize, (list, tuple)):
    raise TypeError(
        "Expected list for 'ksize' argument to "
        "'quantized_avg_pool' Op, not %r." % ksize)
  ksize = [_execute.make_int(_i, "ksize") for _i in ksize]
  if not isinstance(strides, (list, tuple)):
    raise TypeError(
        "Expected list for 'strides' argument to "
        "'quantized_avg_pool' Op, not %r." % strides)
  strides = [_execute.make_int(_i, "strides") for _i in strides]
  padding = _execute.make_str(padding, "padding")
  _attr_T, (input,) = _execute.args_to_matching_eager([input], ctx, [_dtypes.qint8, _dtypes.quint8, _dtypes.qint32, _dtypes.qint16, _dtypes.quint16, ])
  min_input = _ops.convert_to_tensor(min_input, _dtypes.float32)
  max_input = _ops.convert_to_tensor(max_input, _dtypes.float32)
  _inputs_flat = [input, min_input, max_input]
  _attrs = ("T", _attr_T, "ksize", ksize, "strides", strides, "padding",
  padding)
  _result = _execute.execute(b"QuantizedAvgPool", 3, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "QuantizedAvgPool", _inputs_flat, _attrs, _result)
  _result = _QuantizedAvgPoolOutput._make(_result)
  return _result

_QuantizedBatchNormWithGlobalNormalizationOutput = collections.namedtuple(
    "QuantizedBatchNormWithGlobalNormalization",
    ["result", "result_min", "result_max"])


TV_QuantizedBatchNormWithGlobalNormalization_Tinput = TypeVar("TV_QuantizedBatchNormWithGlobalNormalization_Tinput", _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8)
TV_QuantizedBatchNormWithGlobalNormalization_out_type = TypeVar("TV_QuantizedBatchNormWithGlobalNormalization_out_type", _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8)

def quantized_batch_norm_with_global_normalization(t: Annotated[Any, TV_QuantizedBatchNormWithGlobalNormalization_Tinput], t_min: Annotated[Any, _atypes.Float32], t_max: Annotated[Any, _atypes.Float32], m: Annotated[Any, TV_QuantizedBatchNormWithGlobalNormalization_Tinput], m_min: Annotated[Any, _atypes.Float32], m_max: Annotated[Any, _atypes.Float32], v: Annotated[Any, TV_QuantizedBatchNormWithGlobalNormalization_Tinput], v_min: Annotated[Any, _atypes.Float32], v_max: Annotated[Any, _atypes.Float32], beta: Annotated[Any, TV_QuantizedBatchNormWithGlobalNormalization_Tinput], beta_min: Annotated[Any, _atypes.Float32], beta_max: Annotated[Any, _atypes.Float32], gamma: Annotated[Any, TV_QuantizedBatchNormWithGlobalNormalization_Tinput], gamma_min: Annotated[Any, _atypes.Float32], gamma_max: Annotated[Any, _atypes.Float32], out_type: TV_QuantizedBatchNormWithGlobalNormalization_out_type, variance_epsilon: float, scale_after_normalization: bool, name=None):
  r"""Quantized Batch normalization.

  This op is deprecated and will be removed in the future. Prefer
  `tf.nn.batch_normalization`.

  Args:
    t: A `Tensor`. Must be one of the following types: `qint8`, `quint8`, `qint32`, `qint16`, `quint16`.
      A 4D input Tensor.
    t_min: A `Tensor` of type `float32`.
      The value represented by the lowest quantized input.
    t_max: A `Tensor` of type `float32`.
      The value represented by the highest quantized input.
    m: A `Tensor`. Must have the same type as `t`.
      A 1D mean Tensor with size matching the last dimension of t.
      This is the first output from tf.nn.moments,
      or a saved moving average thereof.
    m_min: A `Tensor` of type `float32`.
      The value represented by the lowest quantized mean.
    m_max: A `Tensor` of type `float32`.
      The value represented by the highest quantized mean.
    v: A `Tensor`. Must have the same type as `t`.
      A 1D variance Tensor with size matching the last dimension of t.
      This is the second output from tf.nn.moments,
      or a saved moving average thereof.
    v_min: A `Tensor` of type `float32`.
      The value represented by the lowest quantized variance.
    v_max: A `Tensor` of type `float32`.
      The value represented by the highest quantized variance.
    beta: A `Tensor`. Must have the same type as `t`.
      A 1D beta Tensor with size matching the last dimension of t.
      An offset to be added to the normalized tensor.
    beta_min: A `Tensor` of type `float32`.
      The value represented by the lowest quantized offset.
    beta_max: A `Tensor` of type `float32`.
      The value represented by the highest quantized offset.
    gamma: A `Tensor`. Must have the same type as `t`.
      A 1D gamma Tensor with size matching the last dimension of t.
      If "scale_after_normalization" is true, this tensor will be multiplied
      with the normalized tensor.
    gamma_min: A `Tensor` of type `float32`.
      The value represented by the lowest quantized gamma.
    gamma_max: A `Tensor` of type `float32`.
      The value represented by the highest quantized gamma.
    out_type: A `tf.DType` from: `tf.qint8, tf.quint8, tf.qint32, tf.qint16, tf.quint16`.
    variance_epsilon: A `float`. A small float number to avoid dividing by 0.
    scale_after_normalization: A `bool`.
      A bool indicating whether the resulted tensor
      needs to be multiplied with gamma.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (result, result_min, result_max).

    result: A `Tensor` of type `out_type`.
    result_min: A `Tensor` of type `float32`.
    result_max: A `Tensor` of type `float32`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "QuantizedBatchNormWithGlobalNormalization", name, t, t_min,
        t_max, m, m_min, m_max, v, v_min, v_max, beta, beta_min, beta_max,
        gamma, gamma_min, gamma_max, "out_type", out_type, "variance_epsilon",
        variance_epsilon, "scale_after_normalization",
        scale_after_normalization)
      _result = _QuantizedBatchNormWithGlobalNormalizationOutput._make(_result)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return quantized_batch_norm_with_global_normalization_eager_fallback(
          t, t_min, t_max, m, m_min, m_max, v, v_min, v_max, beta, beta_min,
          beta_max, gamma, gamma_min, gamma_max, out_type=out_type,
          variance_epsilon=variance_epsilon,
          scale_after_normalization=scale_after_normalization, name=name,
          ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  out_type = _execute.make_type(out_type, "out_type")
  variance_epsilon = _execute.make_float(variance_epsilon, "variance_epsilon")
  scale_after_normalization = _execute.make_bool(scale_after_normalization, "scale_after_normalization")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "QuantizedBatchNormWithGlobalNormalization", t=t, t_min=t_min,
                                                     t_max=t_max, m=m,
                                                     m_min=m_min, m_max=m_max,
                                                     v=v, v_min=v_min,
                                                     v_max=v_max, beta=beta,
                                                     beta_min=beta_min,
                                                     beta_max=beta_max,
                                                     gamma=gamma,
                                                     gamma_min=gamma_min,
                                                     gamma_max=gamma_max,
                                                     out_type=out_type,
                                                     variance_epsilon=variance_epsilon,
                                                     scale_after_normalization=scale_after_normalization,
                                                     name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("Tinput", _op._get_attr_type("Tinput"), "out_type",
              _op._get_attr_type("out_type"), "variance_epsilon",
              _op.get_attr("variance_epsilon"), "scale_after_normalization",
              _op._get_attr_bool("scale_after_normalization"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "QuantizedBatchNormWithGlobalNormalization", _inputs_flat, _attrs, _result)
  _result = _QuantizedBatchNormWithGlobalNormalizationOutput._make(_result)
  return _result

QuantizedBatchNormWithGlobalNormalization = tf_export("raw_ops.QuantizedBatchNormWithGlobalNormalization")(_ops.to_raw_op(quantized_batch_norm_with_global_normalization))


def quantized_batch_norm_with_global_normalization_eager_fallback(t: Annotated[Any, TV_QuantizedBatchNormWithGlobalNormalization_Tinput], t_min: Annotated[Any, _atypes.Float32], t_max: Annotated[Any, _atypes.Float32], m: Annotated[Any, TV_QuantizedBatchNormWithGlobalNormalization_Tinput], m_min: Annotated[Any, _atypes.Float32], m_max: Annotated[Any, _atypes.Float32], v: Annotated[Any, TV_QuantizedBatchNormWithGlobalNormalization_Tinput], v_min: Annotated[Any, _atypes.Float32], v_max: Annotated[Any, _atypes.Float32], beta: Annotated[Any, TV_QuantizedBatchNormWithGlobalNormalization_Tinput], beta_min: Annotated[Any, _atypes.Float32], beta_max: Annotated[Any, _atypes.Float32], gamma: Annotated[Any, TV_QuantizedBatchNormWithGlobalNormalization_Tinput], gamma_min: Annotated[Any, _atypes.Float32], gamma_max: Annotated[Any, _atypes.Float32], out_type: TV_QuantizedBatchNormWithGlobalNormalization_out_type, variance_epsilon: float, scale_after_normalization: bool, name, ctx):
  out_type = _execute.make_type(out_type, "out_type")
  variance_epsilon = _execute.make_float(variance_epsilon, "variance_epsilon")
  scale_after_normalization = _execute.make_bool(scale_after_normalization, "scale_after_normalization")
  _attr_Tinput, _inputs_Tinput = _execute.args_to_matching_eager([t, m, v, beta, gamma], ctx, [_dtypes.qint8, _dtypes.quint8, _dtypes.qint32, _dtypes.qint16, _dtypes.quint16, ])
  (t, m, v, beta, gamma) = _inputs_Tinput
  t_min = _ops.convert_to_tensor(t_min, _dtypes.float32)
  t_max = _ops.convert_to_tensor(t_max, _dtypes.float32)
  m_min = _ops.convert_to_tensor(m_min, _dtypes.float32)
  m_max = _ops.convert_to_tensor(m_max, _dtypes.float32)
  v_min = _ops.convert_to_tensor(v_min, _dtypes.float32)
  v_max = _ops.convert_to_tensor(v_max, _dtypes.float32)
  beta_min = _ops.convert_to_tensor(beta_min, _dtypes.float32)
  beta_max = _ops.convert_to_tensor(beta_max, _dtypes.float32)
  gamma_min = _ops.convert_to_tensor(gamma_min, _dtypes.float32)
  gamma_max = _ops.convert_to_tensor(gamma_max, _dtypes.float32)
  _inputs_flat = [t, t_min, t_max, m, m_min, m_max, v, v_min, v_max, beta, beta_min, beta_max, gamma, gamma_min, gamma_max]
  _attrs = ("Tinput", _attr_Tinput, "out_type", out_type, "variance_epsilon",
  variance_epsilon, "scale_after_normalization", scale_after_normalization)
  _result = _execute.execute(b"QuantizedBatchNormWithGlobalNormalization", 3,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "QuantizedBatchNormWithGlobalNormalization", _inputs_flat, _attrs, _result)
  _result = _QuantizedBatchNormWithGlobalNormalizationOutput._make(_result)
  return _result

_QuantizedBiasAddOutput = collections.namedtuple(
    "QuantizedBiasAdd",
    ["output", "min_out", "max_out"])


TV_QuantizedBiasAdd_T1 = TypeVar("TV_QuantizedBiasAdd_T1", _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8)
TV_QuantizedBiasAdd_T2 = TypeVar("TV_QuantizedBiasAdd_T2", _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8)
TV_QuantizedBiasAdd_out_type = TypeVar("TV_QuantizedBiasAdd_out_type", _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8)

def quantized_bias_add(input: Annotated[Any, TV_QuantizedBiasAdd_T1], bias: Annotated[Any, TV_QuantizedBiasAdd_T2], min_input: Annotated[Any, _atypes.Float32], max_input: Annotated[Any, _atypes.Float32], min_bias: Annotated[Any, _atypes.Float32], max_bias: Annotated[Any, _atypes.Float32], out_type: TV_QuantizedBiasAdd_out_type, name=None):
  r"""Adds Tensor 'bias' to Tensor 'input' for Quantized types.

  Broadcasts the values of bias on dimensions 0..N-2 of 'input'.

  Args:
    input: A `Tensor`. Must be one of the following types: `qint8`, `quint8`, `qint32`, `qint16`, `quint16`.
    bias: A `Tensor`. Must be one of the following types: `qint8`, `quint8`, `qint32`, `qint16`, `quint16`.
      A 1D bias Tensor with size matching the last dimension of 'input'.
    min_input: A `Tensor` of type `float32`.
      The float value that the lowest quantized input value represents.
    max_input: A `Tensor` of type `float32`.
      The float value that the highest quantized input value represents.
    min_bias: A `Tensor` of type `float32`.
      The float value that the lowest quantized bias value represents.
    max_bias: A `Tensor` of type `float32`.
      The float value that the highest quantized bias value represents.
    out_type: A `tf.DType` from: `tf.qint8, tf.quint8, tf.qint32, tf.qint16, tf.quint16`.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (output, min_out, max_out).

    output: A `Tensor` of type `out_type`.
    min_out: A `Tensor` of type `float32`.
    max_out: A `Tensor` of type `float32`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "QuantizedBiasAdd", name, input, bias, min_input, max_input,
        min_bias, max_bias, "out_type", out_type)
      _result = _QuantizedBiasAddOutput._make(_result)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return quantized_bias_add_eager_fallback(
          input, bias, min_input, max_input, min_bias, max_bias,
          out_type=out_type, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  out_type = _execute.make_type(out_type, "out_type")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "QuantizedBiasAdd", input=input, bias=bias, min_input=min_input,
                            max_input=max_input, min_bias=min_bias,
                            max_bias=max_bias, out_type=out_type, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T1", _op._get_attr_type("T1"), "T2", _op._get_attr_type("T2"),
              "out_type", _op._get_attr_type("out_type"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "QuantizedBiasAdd", _inputs_flat, _attrs, _result)
  _result = _QuantizedBiasAddOutput._make(_result)
  return _result

QuantizedBiasAdd = tf_export("raw_ops.QuantizedBiasAdd")(_ops.to_raw_op(quantized_bias_add))


def quantized_bias_add_eager_fallback(input: Annotated[Any, TV_QuantizedBiasAdd_T1], bias: Annotated[Any, TV_QuantizedBiasAdd_T2], min_input: Annotated[Any, _atypes.Float32], max_input: Annotated[Any, _atypes.Float32], min_bias: Annotated[Any, _atypes.Float32], max_bias: Annotated[Any, _atypes.Float32], out_type: TV_QuantizedBiasAdd_out_type, name, ctx):
  out_type = _execute.make_type(out_type, "out_type")
  _attr_T1, (input,) = _execute.args_to_matching_eager([input], ctx, [_dtypes.qint8, _dtypes.quint8, _dtypes.qint32, _dtypes.qint16, _dtypes.quint16, ])
  _attr_T2, (bias,) = _execute.args_to_matching_eager([bias], ctx, [_dtypes.qint8, _dtypes.quint8, _dtypes.qint32, _dtypes.qint16, _dtypes.quint16, ])
  min_input = _ops.convert_to_tensor(min_input, _dtypes.float32)
  max_input = _ops.convert_to_tensor(max_input, _dtypes.float32)
  min_bias = _ops.convert_to_tensor(min_bias, _dtypes.float32)
  max_bias = _ops.convert_to_tensor(max_bias, _dtypes.float32)
  _inputs_flat = [input, bias, min_input, max_input, min_bias, max_bias]
  _attrs = ("T1", _attr_T1, "T2", _attr_T2, "out_type", out_type)
  _result = _execute.execute(b"QuantizedBiasAdd", 3, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "QuantizedBiasAdd", _inputs_flat, _attrs, _result)
  _result = _QuantizedBiasAddOutput._make(_result)
  return _result

_QuantizedConv2DOutput = collections.namedtuple(
    "QuantizedConv2D",
    ["output", "min_output", "max_output"])


TV_QuantizedConv2D_Tinput = TypeVar("TV_QuantizedConv2D_Tinput", _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8)
TV_QuantizedConv2D_Tfilter = TypeVar("TV_QuantizedConv2D_Tfilter", _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8)
TV_QuantizedConv2D_out_type = TypeVar("TV_QuantizedConv2D_out_type", _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8)

def quantized_conv2d(input: Annotated[Any, TV_QuantizedConv2D_Tinput], filter: Annotated[Any, TV_QuantizedConv2D_Tfilter], min_input: Annotated[Any, _atypes.Float32], max_input: Annotated[Any, _atypes.Float32], min_filter: Annotated[Any, _atypes.Float32], max_filter: Annotated[Any, _atypes.Float32], strides, padding: str, out_type:TV_QuantizedConv2D_out_type=_dtypes.qint32, dilations=[1, 1, 1, 1], name=None):
  r"""Computes a 2D convolution given quantized 4D input and filter tensors.

  The inputs are quantized tensors where the lowest value represents the real
  number of the associated minimum, and the highest represents the maximum.
  This means that you can only interpret the quantized output in the same way, by
  taking the returned minimum and maximum values into account.

  Args:
    input: A `Tensor`. Must be one of the following types: `qint8`, `quint8`, `qint32`, `qint16`, `quint16`.
    filter: A `Tensor`. Must be one of the following types: `qint8`, `quint8`, `qint32`, `qint16`, `quint16`.
      filter's input_depth dimension must match input's depth dimensions.
    min_input: A `Tensor` of type `float32`.
      The float value that the lowest quantized input value represents.
    max_input: A `Tensor` of type `float32`.
      The float value that the highest quantized input value represents.
    min_filter: A `Tensor` of type `float32`.
      The float value that the lowest quantized filter value represents.
    max_filter: A `Tensor` of type `float32`.
      The float value that the highest quantized filter value represents.
    strides: A list of `ints`.
      The stride of the sliding window for each dimension of the input
      tensor.
    padding: A `string` from: `"SAME", "VALID"`.
      The type of padding algorithm to use.
    out_type: An optional `tf.DType` from: `tf.qint8, tf.quint8, tf.qint32, tf.qint16, tf.quint16`. Defaults to `tf.qint32`.
    dilations: An optional list of `ints`. Defaults to `[1, 1, 1, 1]`.
      1-D tensor of length 4.  The dilation factor for each dimension of
      `input`. If set to k > 1, there will be k-1 skipped cells between each
      filter element on that dimension. The dimension order is determined by the
      value of `data_format`, see above for details. Dilations in the batch and
      depth dimensions must be 1.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (output, min_output, max_output).

    output: A `Tensor` of type `out_type`.
    min_output: A `Tensor` of type `float32`.
    max_output: A `Tensor` of type `float32`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "QuantizedConv2D", name, input, filter, min_input, max_input,
        min_filter, max_filter, "out_type", out_type, "strides", strides,
        "padding", padding, "dilations", dilations)
      _result = _QuantizedConv2DOutput._make(_result)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return quantized_conv2d_eager_fallback(
          input, filter, min_input, max_input, min_filter, max_filter,
          out_type=out_type, strides=strides, padding=padding,
          dilations=dilations, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if not isinstance(strides, (list, tuple)):
    raise TypeError(
        "Expected list for 'strides' argument to "
        "'quantized_conv2d' Op, not %r." % strides)
  strides = [_execute.make_int(_i, "strides") for _i in strides]
  padding = _execute.make_str(padding, "padding")
  if out_type is None:
    out_type = _dtypes.qint32
  out_type = _execute.make_type(out_type, "out_type")
  if dilations is None:
    dilations = [1, 1, 1, 1]
  if not isinstance(dilations, (list, tuple)):
    raise TypeError(
        "Expected list for 'dilations' argument to "
        "'quantized_conv2d' Op, not %r." % dilations)
  dilations = [_execute.make_int(_i, "dilations") for _i in dilations]
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "QuantizedConv2D", input=input, filter=filter, min_input=min_input,
                           max_input=max_input, min_filter=min_filter,
                           max_filter=max_filter, strides=strides,
                           padding=padding, out_type=out_type,
                           dilations=dilations, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("Tinput", _op._get_attr_type("Tinput"), "Tfilter",
              _op._get_attr_type("Tfilter"), "out_type",
              _op._get_attr_type("out_type"), "strides",
              _op.get_attr("strides"), "padding", _op.get_attr("padding"),
              "dilations", _op.get_attr("dilations"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "QuantizedConv2D", _inputs_flat, _attrs, _result)
  _result = _QuantizedConv2DOutput._make(_result)
  return _result

QuantizedConv2D = tf_export("raw_ops.QuantizedConv2D")(_ops.to_raw_op(quantized_conv2d))


def quantized_conv2d_eager_fallback(input: Annotated[Any, TV_QuantizedConv2D_Tinput], filter: Annotated[Any, TV_QuantizedConv2D_Tfilter], min_input: Annotated[Any, _atypes.Float32], max_input: Annotated[Any, _atypes.Float32], min_filter: Annotated[Any, _atypes.Float32], max_filter: Annotated[Any, _atypes.Float32], strides, padding: str, out_type: TV_QuantizedConv2D_out_type, dilations, name, ctx):
  if not isinstance(strides, (list, tuple)):
    raise TypeError(
        "Expected list for 'strides' argument to "
        "'quantized_conv2d' Op, not %r." % strides)
  strides = [_execute.make_int(_i, "strides") for _i in strides]
  padding = _execute.make_str(padding, "padding")
  if out_type is None:
    out_type = _dtypes.qint32
  out_type = _execute.make_type(out_type, "out_type")
  if dilations is None:
    dilations = [1, 1, 1, 1]
  if not isinstance(dilations, (list, tuple)):
    raise TypeError(
        "Expected list for 'dilations' argument to "
        "'quantized_conv2d' Op, not %r." % dilations)
  dilations = [_execute.make_int(_i, "dilations") for _i in dilations]
  _attr_Tinput, (input,) = _execute.args_to_matching_eager([input], ctx, [_dtypes.qint8, _dtypes.quint8, _dtypes.qint32, _dtypes.qint16, _dtypes.quint16, ])
  _attr_Tfilter, (filter,) = _execute.args_to_matching_eager([filter], ctx, [_dtypes.qint8, _dtypes.quint8, _dtypes.qint32, _dtypes.qint16, _dtypes.quint16, ])
  min_input = _ops.convert_to_tensor(min_input, _dtypes.float32)
  max_input = _ops.convert_to_tensor(max_input, _dtypes.float32)
  min_filter = _ops.convert_to_tensor(min_filter, _dtypes.float32)
  max_filter = _ops.convert_to_tensor(max_filter, _dtypes.float32)
  _inputs_flat = [input, filter, min_input, max_input, min_filter, max_filter]
  _attrs = ("Tinput", _attr_Tinput, "Tfilter", _attr_Tfilter, "out_type",
  out_type, "strides", strides, "padding", padding, "dilations", dilations)
  _result = _execute.execute(b"QuantizedConv2D", 3, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "QuantizedConv2D", _inputs_flat, _attrs, _result)
  _result = _QuantizedConv2DOutput._make(_result)
  return _result

_QuantizedConv2DAndReluOutput = collections.namedtuple(
    "QuantizedConv2DAndRelu",
    ["output", "min_output", "max_output"])


TV_QuantizedConv2DAndRelu_Tinput = TypeVar("TV_QuantizedConv2DAndRelu_Tinput", _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8)
TV_QuantizedConv2DAndRelu_Tfilter = TypeVar("TV_QuantizedConv2DAndRelu_Tfilter", _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8)
TV_QuantizedConv2DAndRelu_out_type = TypeVar("TV_QuantizedConv2DAndRelu_out_type", _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8)

def quantized_conv2d_and_relu(input: Annotated[Any, TV_QuantizedConv2DAndRelu_Tinput], filter: Annotated[Any, TV_QuantizedConv2DAndRelu_Tfilter], min_input: Annotated[Any, _atypes.Float32], max_input: Annotated[Any, _atypes.Float32], min_filter: Annotated[Any, _atypes.Float32], max_filter: Annotated[Any, _atypes.Float32], strides, padding: str, out_type:TV_QuantizedConv2DAndRelu_out_type=_dtypes.qint32, dilations=[1, 1, 1, 1], padding_list=[], name=None):
  r"""TODO: add doc.

  Args:
    input: A `Tensor`. Must be one of the following types: `qint8`, `quint8`, `qint32`, `qint16`, `quint16`.
    filter: A `Tensor`. Must be one of the following types: `qint8`, `quint8`, `qint32`, `qint16`, `quint16`.
    min_input: A `Tensor` of type `float32`.
    max_input: A `Tensor` of type `float32`.
    min_filter: A `Tensor` of type `float32`.
    max_filter: A `Tensor` of type `float32`.
    strides: A list of `ints`.
    padding: A `string` from: `"SAME", "VALID"`.
    out_type: An optional `tf.DType` from: `tf.qint8, tf.quint8, tf.qint32, tf.qint16, tf.quint16`. Defaults to `tf.qint32`.
    dilations: An optional list of `ints`. Defaults to `[1, 1, 1, 1]`.
    padding_list: An optional list of `ints`. Defaults to `[]`.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (output, min_output, max_output).

    output: A `Tensor` of type `out_type`.
    min_output: A `Tensor` of type `float32`.
    max_output: A `Tensor` of type `float32`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "QuantizedConv2DAndRelu", name, input, filter, min_input,
        max_input, min_filter, max_filter, "out_type", out_type, "strides",
        strides, "padding", padding, "dilations", dilations, "padding_list",
        padding_list)
      _result = _QuantizedConv2DAndReluOutput._make(_result)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return quantized_conv2d_and_relu_eager_fallback(
          input, filter, min_input, max_input, min_filter, max_filter,
          out_type=out_type, strides=strides, padding=padding,
          dilations=dilations, padding_list=padding_list, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if not isinstance(strides, (list, tuple)):
    raise TypeError(
        "Expected list for 'strides' argument to "
        "'quantized_conv2d_and_relu' Op, not %r." % strides)
  strides = [_execute.make_int(_i, "strides") for _i in strides]
  padding = _execute.make_str(padding, "padding")
  if out_type is None:
    out_type = _dtypes.qint32
  out_type = _execute.make_type(out_type, "out_type")
  if dilations is None:
    dilations = [1, 1, 1, 1]
  if not isinstance(dilations, (list, tuple)):
    raise TypeError(
        "Expected list for 'dilations' argument to "
        "'quantized_conv2d_and_relu' Op, not %r." % dilations)
  dilations = [_execute.make_int(_i, "dilations") for _i in dilations]
  if padding_list is None:
    padding_list = []
  if not isinstance(padding_list, (list, tuple)):
    raise TypeError(
        "Expected list for 'padding_list' argument to "
        "'quantized_conv2d_and_relu' Op, not %r." % padding_list)
  padding_list = [_execute.make_int(_i, "padding_list") for _i in padding_list]
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "QuantizedConv2DAndRelu", input=input, filter=filter,
                                  min_input=min_input, max_input=max_input,
                                  min_filter=min_filter,
                                  max_filter=max_filter, strides=strides,
                                  padding=padding, out_type=out_type,
                                  dilations=dilations,
                                  padding_list=padding_list, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("Tinput", _op._get_attr_type("Tinput"), "Tfilter",
              _op._get_attr_type("Tfilter"), "out_type",
              _op._get_attr_type("out_type"), "strides",
              _op.get_attr("strides"), "padding", _op.get_attr("padding"),
              "dilations", _op.get_attr("dilations"), "padding_list",
              _op.get_attr("padding_list"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "QuantizedConv2DAndRelu", _inputs_flat, _attrs, _result)
  _result = _QuantizedConv2DAndReluOutput._make(_result)
  return _result

QuantizedConv2DAndRelu = tf_export("raw_ops.QuantizedConv2DAndRelu")(_ops.to_raw_op(quantized_conv2d_and_relu))


def quantized_conv2d_and_relu_eager_fallback(input: Annotated[Any, TV_QuantizedConv2DAndRelu_Tinput], filter: Annotated[Any, TV_QuantizedConv2DAndRelu_Tfilter], min_input: Annotated[Any, _atypes.Float32], max_input: Annotated[Any, _atypes.Float32], min_filter: Annotated[Any, _atypes.Float32], max_filter: Annotated[Any, _atypes.Float32], strides, padding: str, out_type: TV_QuantizedConv2DAndRelu_out_type, dilations, padding_list, name, ctx):
  if not isinstance(strides, (list, tuple)):
    raise TypeError(
        "Expected list for 'strides' argument to "
        "'quantized_conv2d_and_relu' Op, not %r." % strides)
  strides = [_execute.make_int(_i, "strides") for _i in strides]
  padding = _execute.make_str(padding, "padding")
  if out_type is None:
    out_type = _dtypes.qint32
  out_type = _execute.make_type(out_type, "out_type")
  if dilations is None:
    dilations = [1, 1, 1, 1]
  if not isinstance(dilations, (list, tuple)):
    raise TypeError(
        "Expected list for 'dilations' argument to "
        "'quantized_conv2d_and_relu' Op, not %r." % dilations)
  dilations = [_execute.make_int(_i, "dilations") for _i in dilations]
  if padding_list is None:
    padding_list = []
  if not isinstance(padding_list, (list, tuple)):
    raise TypeError(
        "Expected list for 'padding_list' argument to "
        "'quantized_conv2d_and_relu' Op, not %r." % padding_list)
  padding_list = [_execute.make_int(_i, "padding_list") for _i in padding_list]
  _attr_Tinput, (input,) = _execute.args_to_matching_eager([input], ctx, [_dtypes.qint8, _dtypes.quint8, _dtypes.qint32, _dtypes.qint16, _dtypes.quint16, ])
  _attr_Tfilter, (filter,) = _execute.args_to_matching_eager([filter], ctx, [_dtypes.qint8, _dtypes.quint8, _dtypes.qint32, _dtypes.qint16, _dtypes.quint16, ])
  min_input = _ops.convert_to_tensor(min_input, _dtypes.float32)
  max_input = _ops.convert_to_tensor(max_input, _dtypes.float32)
  min_filter = _ops.convert_to_tensor(min_filter, _dtypes.float32)
  max_filter = _ops.convert_to_tensor(max_filter, _dtypes.float32)
  _inputs_flat = [input, filter, min_input, max_input, min_filter, max_filter]
  _attrs = ("Tinput", _attr_Tinput, "Tfilter", _attr_Tfilter, "out_type",
  out_type, "strides", strides, "padding", padding, "dilations", dilations,
  "padding_list", padding_list)
  _result = _execute.execute(b"QuantizedConv2DAndRelu", 3,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "QuantizedConv2DAndRelu", _inputs_flat, _attrs, _result)
  _result = _QuantizedConv2DAndReluOutput._make(_result)
  return _result

_QuantizedConv2DAndReluAndRequantizeOutput = collections.namedtuple(
    "QuantizedConv2DAndReluAndRequantize",
    ["output", "min_output", "max_output"])


TV_QuantizedConv2DAndReluAndRequantize_Tinput = TypeVar("TV_QuantizedConv2DAndReluAndRequantize_Tinput", _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8)
TV_QuantizedConv2DAndReluAndRequantize_Tfilter = TypeVar("TV_QuantizedConv2DAndReluAndRequantize_Tfilter", _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8)
TV_QuantizedConv2DAndReluAndRequantize_out_type = TypeVar("TV_QuantizedConv2DAndReluAndRequantize_out_type", _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8)

def quantized_conv2d_and_relu_and_requantize(input: Annotated[Any, TV_QuantizedConv2DAndReluAndRequantize_Tinput], filter: Annotated[Any, TV_QuantizedConv2DAndReluAndRequantize_Tfilter], min_input: Annotated[Any, _atypes.Float32], max_input: Annotated[Any, _atypes.Float32], min_filter: Annotated[Any, _atypes.Float32], max_filter: Annotated[Any, _atypes.Float32], min_freezed_output: Annotated[Any, _atypes.Float32], max_freezed_output: Annotated[Any, _atypes.Float32], strides, padding: str, out_type:TV_QuantizedConv2DAndReluAndRequantize_out_type=_dtypes.quint8, dilations=[1, 1, 1, 1], padding_list=[], name=None):
  r"""TODO: add doc.

  Args:
    input: A `Tensor`. Must be one of the following types: `qint8`, `quint8`, `qint32`, `qint16`, `quint16`.
    filter: A `Tensor`. Must be one of the following types: `qint8`, `quint8`, `qint32`, `qint16`, `quint16`.
    min_input: A `Tensor` of type `float32`.
    max_input: A `Tensor` of type `float32`.
    min_filter: A `Tensor` of type `float32`.
    max_filter: A `Tensor` of type `float32`.
    min_freezed_output: A `Tensor` of type `float32`.
    max_freezed_output: A `Tensor` of type `float32`.
    strides: A list of `ints`.
    padding: A `string` from: `"SAME", "VALID"`.
    out_type: An optional `tf.DType` from: `tf.qint8, tf.quint8, tf.qint32, tf.qint16, tf.quint16`. Defaults to `tf.quint8`.
    dilations: An optional list of `ints`. Defaults to `[1, 1, 1, 1]`.
    padding_list: An optional list of `ints`. Defaults to `[]`.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (output, min_output, max_output).

    output: A `Tensor` of type `out_type`.
    min_output: A `Tensor` of type `float32`.
    max_output: A `Tensor` of type `float32`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "QuantizedConv2DAndReluAndRequantize", name, input, filter,
        min_input, max_input, min_filter, max_filter, min_freezed_output,
        max_freezed_output, "out_type", out_type, "strides", strides,
        "padding", padding, "dilations", dilations, "padding_list",
        padding_list)
      _result = _QuantizedConv2DAndReluAndRequantizeOutput._make(_result)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return quantized_conv2d_and_relu_and_requantize_eager_fallback(
          input, filter, min_input, max_input, min_filter, max_filter,
          min_freezed_output, max_freezed_output, out_type=out_type,
          strides=strides, padding=padding, dilations=dilations,
          padding_list=padding_list, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if not isinstance(strides, (list, tuple)):
    raise TypeError(
        "Expected list for 'strides' argument to "
        "'quantized_conv2d_and_relu_and_requantize' Op, not %r." % strides)
  strides = [_execute.make_int(_i, "strides") for _i in strides]
  padding = _execute.make_str(padding, "padding")
  if out_type is None:
    out_type = _dtypes.quint8
  out_type = _execute.make_type(out_type, "out_type")
  if dilations is None:
    dilations = [1, 1, 1, 1]
  if not isinstance(dilations, (list, tuple)):
    raise TypeError(
        "Expected list for 'dilations' argument to "
        "'quantized_conv2d_and_relu_and_requantize' Op, not %r." % dilations)
  dilations = [_execute.make_int(_i, "dilations") for _i in dilations]
  if padding_list is None:
    padding_list = []
  if not isinstance(padding_list, (list, tuple)):
    raise TypeError(
        "Expected list for 'padding_list' argument to "
        "'quantized_conv2d_and_relu_and_requantize' Op, not %r." % padding_list)
  padding_list = [_execute.make_int(_i, "padding_list") for _i in padding_list]
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "QuantizedConv2DAndReluAndRequantize", input=input, filter=filter,
                                               min_input=min_input,
                                               max_input=max_input,
                                               min_filter=min_filter,
                                               max_filter=max_filter,
                                               min_freezed_output=min_freezed_output,
                                               max_freezed_output=max_freezed_output,
                                               strides=strides,
                                               padding=padding,
                                               out_type=out_type,
                                               dilations=dilations,
                                               padding_list=padding_list,
                                               name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("Tinput", _op._get_attr_type("Tinput"), "Tfilter",
              _op._get_attr_type("Tfilter"), "out_type",
              _op._get_attr_type("out_type"), "strides",
              _op.get_attr("strides"), "padding", _op.get_attr("padding"),
              "dilations", _op.get_attr("dilations"), "padding_list",
              _op.get_attr("padding_list"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "QuantizedConv2DAndReluAndRequantize", _inputs_flat, _attrs, _result)
  _result = _QuantizedConv2DAndReluAndRequantizeOutput._make(_result)
  return _result

QuantizedConv2DAndReluAndRequantize = tf_export("raw_ops.QuantizedConv2DAndReluAndRequantize")(_ops.to_raw_op(quantized_conv2d_and_relu_and_requantize))


def quantized_conv2d_and_relu_and_requantize_eager_fallback(input: Annotated[Any, TV_QuantizedConv2DAndReluAndRequantize_Tinput], filter: Annotated[Any, TV_QuantizedConv2DAndReluAndRequantize_Tfilter], min_input: Annotated[Any, _atypes.Float32], max_input: Annotated[Any, _atypes.Float32], min_filter: Annotated[Any, _atypes.Float32], max_filter: Annotated[Any, _atypes.Float32], min_freezed_output: Annotated[Any, _atypes.Float32], max_freezed_output: Annotated[Any, _atypes.Float32], strides, padding: str, out_type: TV_QuantizedConv2DAndReluAndRequantize_out_type, dilations, padding_list, name, ctx):
  if not isinstance(strides, (list, tuple)):
    raise TypeError(
        "Expected list for 'strides' argument to "
        "'quantized_conv2d_and_relu_and_requantize' Op, not %r." % strides)
  strides = [_execute.make_int(_i, "strides") for _i in strides]
  padding = _execute.make_str(padding, "padding")
  if out_type is None:
    out_type = _dtypes.quint8
  out_type = _execute.make_type(out_type, "out_type")
  if dilations is None:
    dilations = [1, 1, 1, 1]
  if not isinstance(dilations, (list, tuple)):
    raise TypeError(
        "Expected list for 'dilations' argument to "
        "'quantized_conv2d_and_relu_and_requantize' Op, not %r." % dilations)
  dilations = [_execute.make_int(_i, "dilations") for _i in dilations]
  if padding_list is None:
    padding_list = []
  if not isinstance(padding_list, (list, tuple)):
    raise TypeError(
        "Expected list for 'padding_list' argument to "
        "'quantized_conv2d_and_relu_and_requantize' Op, not %r." % padding_list)
  padding_list = [_execute.make_int(_i, "padding_list") for _i in padding_list]
  _attr_Tinput, (input,) = _execute.args_to_matching_eager([input], ctx, [_dtypes.qint8, _dtypes.quint8, _dtypes.qint32, _dtypes.qint16, _dtypes.quint16, ])
  _attr_Tfilter, (filter,) = _execute.args_to_matching_eager([filter], ctx, [_dtypes.qint8, _dtypes.quint8, _dtypes.qint32, _dtypes.qint16, _dtypes.quint16, ])
  min_input = _ops.convert_to_tensor(min_input, _dtypes.float32)
  max_input = _ops.convert_to_tensor(max_input, _dtypes.float32)
  min_filter = _ops.convert_to_tensor(min_filter, _dtypes.float32)
  max_filter = _ops.convert_to_tensor(max_filter, _dtypes.float32)
  min_freezed_output = _ops.convert_to_tensor(min_freezed_output, _dtypes.float32)
  max_freezed_output = _ops.convert_to_tensor(max_freezed_output, _dtypes.float32)
  _inputs_flat = [input, filter, min_input, max_input, min_filter, max_filter, min_freezed_output, max_freezed_output]
  _attrs = ("Tinput", _attr_Tinput, "Tfilter", _attr_Tfilter, "out_type",
  out_type, "strides", strides, "padding", padding, "dilations", dilations,
  "padding_list", padding_list)
  _result = _execute.execute(b"QuantizedConv2DAndReluAndRequantize", 3,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "QuantizedConv2DAndReluAndRequantize", _inputs_flat, _attrs, _result)
  _result = _QuantizedConv2DAndReluAndRequantizeOutput._make(_result)
  return _result

_QuantizedConv2DAndRequantizeOutput = collections.namedtuple(
    "QuantizedConv2DAndRequantize",
    ["output", "min_output", "max_output"])


TV_QuantizedConv2DAndRequantize_Tinput = TypeVar("TV_QuantizedConv2DAndRequantize_Tinput", _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8)
TV_QuantizedConv2DAndRequantize_Tfilter = TypeVar("TV_QuantizedConv2DAndRequantize_Tfilter", _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8)
TV_QuantizedConv2DAndRequantize_out_type = TypeVar("TV_QuantizedConv2DAndRequantize_out_type", _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8)

def quantized_conv2d_and_requantize(input: Annotated[Any, TV_QuantizedConv2DAndRequantize_Tinput], filter: Annotated[Any, TV_QuantizedConv2DAndRequantize_Tfilter], min_input: Annotated[Any, _atypes.Float32], max_input: Annotated[Any, _atypes.Float32], min_filter: Annotated[Any, _atypes.Float32], max_filter: Annotated[Any, _atypes.Float32], min_freezed_output: Annotated[Any, _atypes.Float32], max_freezed_output: Annotated[Any, _atypes.Float32], strides, padding: str, out_type:TV_QuantizedConv2DAndRequantize_out_type=_dtypes.qint8, dilations=[1, 1, 1, 1], padding_list=[], name=None):
  r"""TODO: add doc.

  Args:
    input: A `Tensor`. Must be one of the following types: `qint8`, `quint8`, `qint32`, `qint16`, `quint16`.
    filter: A `Tensor`. Must be one of the following types: `qint8`, `quint8`, `qint32`, `qint16`, `quint16`.
    min_input: A `Tensor` of type `float32`.
    max_input: A `Tensor` of type `float32`.
    min_filter: A `Tensor` of type `float32`.
    max_filter: A `Tensor` of type `float32`.
    min_freezed_output: A `Tensor` of type `float32`.
    max_freezed_output: A `Tensor` of type `float32`.
    strides: A list of `ints`.
    padding: A `string` from: `"SAME", "VALID"`.
    out_type: An optional `tf.DType` from: `tf.qint8, tf.quint8, tf.qint32, tf.qint16, tf.quint16`. Defaults to `tf.qint8`.
    dilations: An optional list of `ints`. Defaults to `[1, 1, 1, 1]`.
    padding_list: An optional list of `ints`. Defaults to `[]`.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (output, min_output, max_output).

    output: A `Tensor` of type `out_type`.
    min_output: A `Tensor` of type `float32`.
    max_output: A `Tensor` of type `float32`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "QuantizedConv2DAndRequantize", name, input, filter, min_input,
        max_input, min_filter, max_filter, min_freezed_output,
        max_freezed_output, "out_type", out_type, "strides", strides,
        "padding", padding, "dilations", dilations, "padding_list",
        padding_list)
      _result = _QuantizedConv2DAndRequantizeOutput._make(_result)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return quantized_conv2d_and_requantize_eager_fallback(
          input, filter, min_input, max_input, min_filter, max_filter,
          min_freezed_output, max_freezed_output, out_type=out_type,
          strides=strides, padding=padding, dilations=dilations,
          padding_list=padding_list, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if not isinstance(strides, (list, tuple)):
    raise TypeError(
        "Expected list for 'strides' argument to "
        "'quantized_conv2d_and_requantize' Op, not %r." % strides)
  strides = [_execute.make_int(_i, "strides") for _i in strides]
  padding = _execute.make_str(padding, "padding")
  if out_type is None:
    out_type = _dtypes.qint8
  out_type = _execute.make_type(out_type, "out_type")
  if dilations is None:
    dilations = [1, 1, 1, 1]
  if not isinstance(dilations, (list, tuple)):
    raise TypeError(
        "Expected list for 'dilations' argument to "
        "'quantized_conv2d_and_requantize' Op, not %r." % dilations)
  dilations = [_execute.make_int(_i, "dilations") for _i in dilations]
  if padding_list is None:
    padding_list = []
  if not isinstance(padding_list, (list, tuple)):
    raise TypeError(
        "Expected list for 'padding_list' argument to "
        "'quantized_conv2d_and_requantize' Op, not %r." % padding_list)
  padding_list = [_execute.make_int(_i, "padding_list") for _i in padding_list]
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "QuantizedConv2DAndRequantize", input=input, filter=filter,
                                        min_input=min_input,
                                        max_input=max_input,
                                        min_filter=min_filter,
                                        max_filter=max_filter,
                                        min_freezed_output=min_freezed_output,
                                        max_freezed_output=max_freezed_output,
                                        strides=strides, padding=padding,
                                        out_type=out_type,
                                        dilations=dilations,
                                        padding_list=padding_list, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("Tinput", _op._get_attr_type("Tinput"), "Tfilter",
              _op._get_attr_type("Tfilter"), "out_type",
              _op._get_attr_type("out_type"), "strides",
              _op.get_attr("strides"), "padding", _op.get_attr("padding"),
              "dilations", _op.get_attr("dilations"), "padding_list",
              _op.get_attr("padding_list"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "QuantizedConv2DAndRequantize", _inputs_flat, _attrs, _result)
  _result = _QuantizedConv2DAndRequantizeOutput._make(_result)
  return _result

QuantizedConv2DAndRequantize = tf_export("raw_ops.QuantizedConv2DAndRequantize")(_ops.to_raw_op(quantized_conv2d_and_requantize))


def quantized_conv2d_and_requantize_eager_fallback(input: Annotated[Any, TV_QuantizedConv2DAndRequantize_Tinput], filter: Annotated[Any, TV_QuantizedConv2DAndRequantize_Tfilter], min_input: Annotated[Any, _atypes.Float32], max_input: Annotated[Any, _atypes.Float32], min_filter: Annotated[Any, _atypes.Float32], max_filter: Annotated[Any, _atypes.Float32], min_freezed_output: Annotated[Any, _atypes.Float32], max_freezed_output: Annotated[Any, _atypes.Float32], strides, padding: str, out_type: TV_QuantizedConv2DAndRequantize_out_type, dilations, padding_list, name, ctx):
  if not isinstance(strides, (list, tuple)):
    raise TypeError(
        "Expected list for 'strides' argument to "
        "'quantized_conv2d_and_requantize' Op, not %r." % strides)
  strides = [_execute.make_int(_i, "strides") for _i in strides]
  padding = _execute.make_str(padding, "padding")
  if out_type is None:
    out_type = _dtypes.qint8
  out_type = _execute.make_type(out_type, "out_type")
  if dilations is None:
    dilations = [1, 1, 1, 1]
  if not isinstance(dilations, (list, tuple)):
    raise TypeError(
        "Expected list for 'dilations' argument to "
        "'quantized_conv2d_and_requantize' Op, not %r." % dilations)
  dilations = [_execute.make_int(_i, "dilations") for _i in dilations]
  if padding_list is None:
    padding_list = []
  if not isinstance(padding_list, (list, tuple)):
    raise TypeError(
        "Expected list for 'padding_list' argument to "
        "'quantized_conv2d_and_requantize' Op, not %r." % padding_list)
  padding_list = [_execute.make_int(_i, "padding_list") for _i in padding_list]
  _attr_Tinput, (input,) = _execute.args_to_matching_eager([input], ctx, [_dtypes.qint8, _dtypes.quint8, _dtypes.qint32, _dtypes.qint16, _dtypes.quint16, ])
  _attr_Tfilter, (filter,) = _execute.args_to_matching_eager([filter], ctx, [_dtypes.qint8, _dtypes.quint8, _dtypes.qint32, _dtypes.qint16, _dtypes.quint16, ])
  min_input = _ops.convert_to_tensor(min_input, _dtypes.float32)
  max_input = _ops.convert_to_tensor(max_input, _dtypes.float32)
  min_filter = _ops.convert_to_tensor(min_filter, _dtypes.float32)
  max_filter = _ops.convert_to_tensor(max_filter, _dtypes.float32)
  min_freezed_output = _ops.convert_to_tensor(min_freezed_output, _dtypes.float32)
  max_freezed_output = _ops.convert_to_tensor(max_freezed_output, _dtypes.float32)
  _inputs_flat = [input, filter, min_input, max_input, min_filter, max_filter, min_freezed_output, max_freezed_output]
  _attrs = ("Tinput", _attr_Tinput, "Tfilter", _attr_Tfilter, "out_type",
  out_type, "strides", strides, "padding", padding, "dilations", dilations,
  "padding_list", padding_list)
  _result = _execute.execute(b"QuantizedConv2DAndRequantize", 3,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "QuantizedConv2DAndRequantize", _inputs_flat, _attrs, _result)
  _result = _QuantizedConv2DAndRequantizeOutput._make(_result)
  return _result

_QuantizedConv2DPerChannelOutput = collections.namedtuple(
    "QuantizedConv2DPerChannel",
    ["output", "min_output", "max_output"])


TV_QuantizedConv2DPerChannel_Tinput = TypeVar("TV_QuantizedConv2DPerChannel_Tinput", _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8)
TV_QuantizedConv2DPerChannel_Tfilter = TypeVar("TV_QuantizedConv2DPerChannel_Tfilter", _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8)
TV_QuantizedConv2DPerChannel_out_type = TypeVar("TV_QuantizedConv2DPerChannel_out_type", _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8)

def quantized_conv2d_per_channel(input: Annotated[Any, TV_QuantizedConv2DPerChannel_Tinput], filter: Annotated[Any, TV_QuantizedConv2DPerChannel_Tfilter], min_input: Annotated[Any, _atypes.Float32], max_input: Annotated[Any, _atypes.Float32], min_filter: Annotated[Any, _atypes.Float32], max_filter: Annotated[Any, _atypes.Float32], strides, padding: str, out_type:TV_QuantizedConv2DPerChannel_out_type=_dtypes.qint32, dilations=[1, 1, 1, 1], name=None):
  r"""Computes QuantizedConv2D per channel.

  Args:
    input: A `Tensor`. Must be one of the following types: `qint8`, `quint8`, `qint32`, `qint16`, `quint16`.
      The original input tensor.
    filter: A `Tensor`. Must be one of the following types: `qint8`, `quint8`, `qint32`, `qint16`, `quint16`.
      The original filter tensor.
    min_input: A `Tensor` of type `float32`.
      The minimum value of the input tensor
    max_input: A `Tensor` of type `float32`.
      The maximum value of the input tensor.
    min_filter: A `Tensor` of type `float32`.
      The minimum value of the filter tensor.
    max_filter: A `Tensor` of type `float32`.
      The maximum value of the filter tensor.
    strides: A list of `ints`. list of stride values.
    padding: A `string` from: `"SAME", "VALID"`.
    out_type: An optional `tf.DType` from: `tf.qint8, tf.quint8, tf.qint32, tf.qint16, tf.quint16`. Defaults to `tf.qint32`.
      The quantized type of output tensor that needs to be converted.
    dilations: An optional list of `ints`. Defaults to `[1, 1, 1, 1]`.
      list of dilation values.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (output, min_output, max_output).

    output: A `Tensor` of type `out_type`.
    min_output: A `Tensor` of type `float32`.
    max_output: A `Tensor` of type `float32`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "QuantizedConv2DPerChannel", name, input, filter, min_input,
        max_input, min_filter, max_filter, "out_type", out_type, "strides",
        strides, "padding", padding, "dilations", dilations)
      _result = _QuantizedConv2DPerChannelOutput._make(_result)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return quantized_conv2d_per_channel_eager_fallback(
          input, filter, min_input, max_input, min_filter, max_filter,
          out_type=out_type, strides=strides, padding=padding,
          dilations=dilations, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if not isinstance(strides, (list, tuple)):
    raise TypeError(
        "Expected list for 'strides' argument to "
        "'quantized_conv2d_per_channel' Op, not %r." % strides)
  strides = [_execute.make_int(_i, "strides") for _i in strides]
  padding = _execute.make_str(padding, "padding")
  if out_type is None:
    out_type = _dtypes.qint32
  out_type = _execute.make_type(out_type, "out_type")
  if dilations is None:
    dilations = [1, 1, 1, 1]
  if not isinstance(dilations, (list, tuple)):
    raise TypeError(
        "Expected list for 'dilations' argument to "
        "'quantized_conv2d_per_channel' Op, not %r." % dilations)
  dilations = [_execute.make_int(_i, "dilations") for _i in dilations]
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "QuantizedConv2DPerChannel", input=input, filter=filter,
                                     min_input=min_input, max_input=max_input,
                                     min_filter=min_filter,
                                     max_filter=max_filter, strides=strides,
                                     padding=padding, out_type=out_type,
                                     dilations=dilations, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("Tinput", _op._get_attr_type("Tinput"), "Tfilter",
              _op._get_attr_type("Tfilter"), "out_type",
              _op._get_attr_type("out_type"), "strides",
              _op.get_attr("strides"), "padding", _op.get_attr("padding"),
              "dilations", _op.get_attr("dilations"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "QuantizedConv2DPerChannel", _inputs_flat, _attrs, _result)
  _result = _QuantizedConv2DPerChannelOutput._make(_result)
  return _result

QuantizedConv2DPerChannel = tf_export("raw_ops.QuantizedConv2DPerChannel")(_ops.to_raw_op(quantized_conv2d_per_channel))


def quantized_conv2d_per_channel_eager_fallback(input: Annotated[Any, TV_QuantizedConv2DPerChannel_Tinput], filter: Annotated[Any, TV_QuantizedConv2DPerChannel_Tfilter], min_input: Annotated[Any, _atypes.Float32], max_input: Annotated[Any, _atypes.Float32], min_filter: Annotated[Any, _atypes.Float32], max_filter: Annotated[Any, _atypes.Float32], strides, padding: str, out_type: TV_QuantizedConv2DPerChannel_out_type, dilations, name, ctx):
  if not isinstance(strides, (list, tuple)):
    raise TypeError(
        "Expected list for 'strides' argument to "
        "'quantized_conv2d_per_channel' Op, not %r." % strides)
  strides = [_execute.make_int(_i, "strides") for _i in strides]
  padding = _execute.make_str(padding, "padding")
  if out_type is None:
    out_type = _dtypes.qint32
  out_type = _execute.make_type(out_type, "out_type")
  if dilations is None:
    dilations = [1, 1, 1, 1]
  if not isinstance(dilations, (list, tuple)):
    raise TypeError(
        "Expected list for 'dilations' argument to "
        "'quantized_conv2d_per_channel' Op, not %r." % dilations)
  dilations = [_execute.make_int(_i, "dilations") for _i in dilations]
  _attr_Tinput, (input,) = _execute.args_to_matching_eager([input], ctx, [_dtypes.qint8, _dtypes.quint8, _dtypes.qint32, _dtypes.qint16, _dtypes.quint16, ])
  _attr_Tfilter, (filter,) = _execute.args_to_matching_eager([filter], ctx, [_dtypes.qint8, _dtypes.quint8, _dtypes.qint32, _dtypes.qint16, _dtypes.quint16, ])
  min_input = _ops.convert_to_tensor(min_input, _dtypes.float32)
  max_input = _ops.convert_to_tensor(max_input, _dtypes.float32)
  min_filter = _ops.convert_to_tensor(min_filter, _dtypes.float32)
  max_filter = _ops.convert_to_tensor(max_filter, _dtypes.float32)
  _inputs_flat = [input, filter, min_input, max_input, min_filter, max_filter]
  _attrs = ("Tinput", _attr_Tinput, "Tfilter", _attr_Tfilter, "out_type",
  out_type, "strides", strides, "padding", padding, "dilations", dilations)
  _result = _execute.execute(b"QuantizedConv2DPerChannel", 3,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "QuantizedConv2DPerChannel", _inputs_flat, _attrs, _result)
  _result = _QuantizedConv2DPerChannelOutput._make(_result)
  return _result

_QuantizedConv2DWithBiasOutput = collections.namedtuple(
    "QuantizedConv2DWithBias",
    ["output", "min_output", "max_output"])


TV_QuantizedConv2DWithBias_Tinput = TypeVar("TV_QuantizedConv2DWithBias_Tinput", _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8)
TV_QuantizedConv2DWithBias_Tfilter = TypeVar("TV_QuantizedConv2DWithBias_Tfilter", _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8)
TV_QuantizedConv2DWithBias_out_type = TypeVar("TV_QuantizedConv2DWithBias_out_type", _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8)

def quantized_conv2d_with_bias(input: Annotated[Any, TV_QuantizedConv2DWithBias_Tinput], filter: Annotated[Any, TV_QuantizedConv2DWithBias_Tfilter], bias: Annotated[Any, _atypes.Float32], min_input: Annotated[Any, _atypes.Float32], max_input: Annotated[Any, _atypes.Float32], min_filter: Annotated[Any, _atypes.Float32], max_filter: Annotated[Any, _atypes.Float32], strides, padding: str, out_type:TV_QuantizedConv2DWithBias_out_type=_dtypes.qint32, dilations=[1, 1, 1, 1], padding_list=[], name=None):
  r"""TODO: add doc.

  Args:
    input: A `Tensor`. Must be one of the following types: `qint8`, `quint8`, `qint32`, `qint16`, `quint16`.
    filter: A `Tensor`. Must be one of the following types: `qint8`, `quint8`, `qint32`, `qint16`, `quint16`.
    bias: A `Tensor` of type `float32`.
    min_input: A `Tensor` of type `float32`.
    max_input: A `Tensor` of type `float32`.
    min_filter: A `Tensor` of type `float32`.
    max_filter: A `Tensor` of type `float32`.
    strides: A list of `ints`.
    padding: A `string` from: `"SAME", "VALID"`.
    out_type: An optional `tf.DType` from: `tf.qint8, tf.quint8, tf.qint32, tf.qint16, tf.quint16`. Defaults to `tf.qint32`.
    dilations: An optional list of `ints`. Defaults to `[1, 1, 1, 1]`.
    padding_list: An optional list of `ints`. Defaults to `[]`.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (output, min_output, max_output).

    output: A `Tensor` of type `out_type`.
    min_output: A `Tensor` of type `float32`.
    max_output: A `Tensor` of type `float32`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "QuantizedConv2DWithBias", name, input, filter, bias, min_input,
        max_input, min_filter, max_filter, "out_type", out_type, "strides",
        strides, "padding", padding, "dilations", dilations, "padding_list",
        padding_list)
      _result = _QuantizedConv2DWithBiasOutput._make(_result)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return quantized_conv2d_with_bias_eager_fallback(
          input, filter, bias, min_input, max_input, min_filter, max_filter,
          out_type=out_type, strides=strides, padding=padding,
          dilations=dilations, padding_list=padding_list, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if not isinstance(strides, (list, tuple)):
    raise TypeError(
        "Expected list for 'strides' argument to "
        "'quantized_conv2d_with_bias' Op, not %r." % strides)
  strides = [_execute.make_int(_i, "strides") for _i in strides]
  padding = _execute.make_str(padding, "padding")
  if out_type is None:
    out_type = _dtypes.qint32
  out_type = _execute.make_type(out_type, "out_type")
  if dilations is None:
    dilations = [1, 1, 1, 1]
  if not isinstance(dilations, (list, tuple)):
    raise TypeError(
        "Expected list for 'dilations' argument to "
        "'quantized_conv2d_with_bias' Op, not %r." % dilations)
  dilations = [_execute.make_int(_i, "dilations") for _i in dilations]
  if padding_list is None:
    padding_list = []
  if not isinstance(padding_list, (list, tuple)):
    raise TypeError(
        "Expected list for 'padding_list' argument to "
        "'quantized_conv2d_with_bias' Op, not %r." % padding_list)
  padding_list = [_execute.make_int(_i, "padding_list") for _i in padding_list]
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "QuantizedConv2DWithBias", input=input, filter=filter, bias=bias,
                                   min_input=min_input, max_input=max_input,
                                   min_filter=min_filter,
                                   max_filter=max_filter, strides=strides,
                                   padding=padding, out_type=out_type,
                                   dilations=dilations,
                                   padding_list=padding_list, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("Tinput", _op._get_attr_type("Tinput"), "Tfilter",
              _op._get_attr_type("Tfilter"), "out_type",
              _op._get_attr_type("out_type"), "strides",
              _op.get_attr("strides"), "padding", _op.get_attr("padding"),
              "dilations", _op.get_attr("dilations"), "padding_list",
              _op.get_attr("padding_list"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "QuantizedConv2DWithBias", _inputs_flat, _attrs, _result)
  _result = _QuantizedConv2DWithBiasOutput._make(_result)
  return _result

QuantizedConv2DWithBias = tf_export("raw_ops.QuantizedConv2DWithBias")(_ops.to_raw_op(quantized_conv2d_with_bias))


def quantized_conv2d_with_bias_eager_fallback(input: Annotated[Any, TV_QuantizedConv2DWithBias_Tinput], filter: Annotated[Any, TV_QuantizedConv2DWithBias_Tfilter], bias: Annotated[Any, _atypes.Float32], min_input: Annotated[Any, _atypes.Float32], max_input: Annotated[Any, _atypes.Float32], min_filter: Annotated[Any, _atypes.Float32], max_filter: Annotated[Any, _atypes.Float32], strides, padding: str, out_type: TV_QuantizedConv2DWithBias_out_type, dilations, padding_list, name, ctx):
  if not isinstance(strides, (list, tuple)):
    raise TypeError(
        "Expected list for 'strides' argument to "
        "'quantized_conv2d_with_bias' Op, not %r." % strides)
  strides = [_execute.make_int(_i, "strides") for _i in strides]
  padding = _execute.make_str(padding, "padding")
  if out_type is None:
    out_type = _dtypes.qint32
  out_type = _execute.make_type(out_type, "out_type")
  if dilations is None:
    dilations = [1, 1, 1, 1]
  if not isinstance(dilations, (list, tuple)):
    raise TypeError(
        "Expected list for 'dilations' argument to "
        "'quantized_conv2d_with_bias' Op, not %r." % dilations)
  dilations = [_execute.make_int(_i, "dilations") for _i in dilations]
  if padding_list is None:
    padding_list = []
  if not isinstance(padding_list, (list, tuple)):
    raise TypeError(
        "Expected list for 'padding_list' argument to "
        "'quantized_conv2d_with_bias' Op, not %r." % padding_list)
  padding_list = [_execute.make_int(_i, "padding_list") for _i in padding_list]
  _attr_Tinput, (input,) = _execute.args_to_matching_eager([input], ctx, [_dtypes.qint8, _dtypes.quint8, _dtypes.qint32, _dtypes.qint16, _dtypes.quint16, ])
  _attr_Tfilter, (filter,) = _execute.args_to_matching_eager([filter], ctx, [_dtypes.qint8, _dtypes.quint8, _dtypes.qint32, _dtypes.qint16, _dtypes.quint16, ])
  bias = _ops.convert_to_tensor(bias, _dtypes.float32)
  min_input = _ops.convert_to_tensor(min_input, _dtypes.float32)
  max_input = _ops.convert_to_tensor(max_input, _dtypes.float32)
  min_filter = _ops.convert_to_tensor(min_filter, _dtypes.float32)
  max_filter = _ops.convert_to_tensor(max_filter, _dtypes.float32)
  _inputs_flat = [input, filter, bias, min_input, max_input, min_filter, max_filter]
  _attrs = ("Tinput", _attr_Tinput, "Tfilter", _attr_Tfilter, "out_type",
  out_type, "strides", strides, "padding", padding, "dilations", dilations,
  "padding_list", padding_list)
  _result = _execute.execute(b"QuantizedConv2DWithBias", 3,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "QuantizedConv2DWithBias", _inputs_flat, _attrs, _result)
  _result = _QuantizedConv2DWithBiasOutput._make(_result)
  return _result

_QuantizedConv2DWithBiasAndReluOutput = collections.namedtuple(
    "QuantizedConv2DWithBiasAndRelu",
    ["output", "min_output", "max_output"])


TV_QuantizedConv2DWithBiasAndRelu_Tinput = TypeVar("TV_QuantizedConv2DWithBiasAndRelu_Tinput", _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8)
TV_QuantizedConv2DWithBiasAndRelu_Tfilter = TypeVar("TV_QuantizedConv2DWithBiasAndRelu_Tfilter", _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8)
TV_QuantizedConv2DWithBiasAndRelu_out_type = TypeVar("TV_QuantizedConv2DWithBiasAndRelu_out_type", _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8)

def quantized_conv2d_with_bias_and_relu(input: Annotated[Any, TV_QuantizedConv2DWithBiasAndRelu_Tinput], filter: Annotated[Any, TV_QuantizedConv2DWithBiasAndRelu_Tfilter], bias: Annotated[Any, _atypes.Float32], min_input: Annotated[Any, _atypes.Float32], max_input: Annotated[Any, _atypes.Float32], min_filter: Annotated[Any, _atypes.Float32], max_filter: Annotated[Any, _atypes.Float32], strides, padding: str, out_type:TV_QuantizedConv2DWithBiasAndRelu_out_type=_dtypes.qint32, dilations=[1, 1, 1, 1], padding_list=[], name=None):
  r"""TODO: add doc.

  Args:
    input: A `Tensor`. Must be one of the following types: `qint8`, `quint8`, `qint32`, `qint16`, `quint16`.
    filter: A `Tensor`. Must be one of the following types: `qint8`, `quint8`, `qint32`, `qint16`, `quint16`.
    bias: A `Tensor` of type `float32`.
    min_input: A `Tensor` of type `float32`.
    max_input: A `Tensor` of type `float32`.
    min_filter: A `Tensor` of type `float32`.
    max_filter: A `Tensor` of type `float32`.
    strides: A list of `ints`.
    padding: A `string` from: `"SAME", "VALID"`.
    out_type: An optional `tf.DType` from: `tf.qint8, tf.quint8, tf.qint32, tf.qint16, tf.quint16`. Defaults to `tf.qint32`.
    dilations: An optional list of `ints`. Defaults to `[1, 1, 1, 1]`.
    padding_list: An optional list of `ints`. Defaults to `[]`.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (output, min_output, max_output).

    output: A `Tensor` of type `out_type`.
    min_output: A `Tensor` of type `float32`.
    max_output: A `Tensor` of type `float32`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "QuantizedConv2DWithBiasAndRelu", name, input, filter, bias,
        min_input, max_input, min_filter, max_filter, "out_type", out_type,
        "strides", strides, "padding", padding, "dilations", dilations,
        "padding_list", padding_list)
      _result = _QuantizedConv2DWithBiasAndReluOutput._make(_result)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return quantized_conv2d_with_bias_and_relu_eager_fallback(
          input, filter, bias, min_input, max_input, min_filter, max_filter,
          out_type=out_type, strides=strides, padding=padding,
          dilations=dilations, padding_list=padding_list, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if not isinstance(strides, (list, tuple)):
    raise TypeError(
        "Expected list for 'strides' argument to "
        "'quantized_conv2d_with_bias_and_relu' Op, not %r." % strides)
  strides = [_execute.make_int(_i, "strides") for _i in strides]
  padding = _execute.make_str(padding, "padding")
  if out_type is None:
    out_type = _dtypes.qint32
  out_type = _execute.make_type(out_type, "out_type")
  if dilations is None:
    dilations = [1, 1, 1, 1]
  if not isinstance(dilations, (list, tuple)):
    raise TypeError(
        "Expected list for 'dilations' argument to "
        "'quantized_conv2d_with_bias_and_relu' Op, not %r." % dilations)
  dilations = [_execute.make_int(_i, "dilations") for _i in dilations]
  if padding_list is None:
    padding_list = []
  if not isinstance(padding_list, (list, tuple)):
    raise TypeError(
        "Expected list for 'padding_list' argument to "
        "'quantized_conv2d_with_bias_and_relu' Op, not %r." % padding_list)
  padding_list = [_execute.make_int(_i, "padding_list") for _i in padding_list]
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "QuantizedConv2DWithBiasAndRelu", input=input, filter=filter,
                                          bias=bias, min_input=min_input,
                                          max_input=max_input,
                                          min_filter=min_filter,
                                          max_filter=max_filter,
                                          strides=strides, padding=padding,
                                          out_type=out_type,
                                          dilations=dilations,
                                          padding_list=padding_list,
                                          name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("Tinput", _op._get_attr_type("Tinput"), "Tfilter",
              _op._get_attr_type("Tfilter"), "out_type",
              _op._get_attr_type("out_type"), "strides",
              _op.get_attr("strides"), "padding", _op.get_attr("padding"),
              "dilations", _op.get_attr("dilations"), "padding_list",
              _op.get_attr("padding_list"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "QuantizedConv2DWithBiasAndRelu", _inputs_flat, _attrs, _result)
  _result = _QuantizedConv2DWithBiasAndReluOutput._make(_result)
  return _result

QuantizedConv2DWithBiasAndRelu = tf_export("raw_ops.QuantizedConv2DWithBiasAndRelu")(_ops.to_raw_op(quantized_conv2d_with_bias_and_relu))


def quantized_conv2d_with_bias_and_relu_eager_fallback(input: Annotated[Any, TV_QuantizedConv2DWithBiasAndRelu_Tinput], filter: Annotated[Any, TV_QuantizedConv2DWithBiasAndRelu_Tfilter], bias: Annotated[Any, _atypes.Float32], min_input: Annotated[Any, _atypes.Float32], max_input: Annotated[Any, _atypes.Float32], min_filter: Annotated[Any, _atypes.Float32], max_filter: Annotated[Any, _atypes.Float32], strides, padding: str, out_type: TV_QuantizedConv2DWithBiasAndRelu_out_type, dilations, padding_list, name, ctx):
  if not isinstance(strides, (list, tuple)):
    raise TypeError(
        "Expected list for 'strides' argument to "
        "'quantized_conv2d_with_bias_and_relu' Op, not %r." % strides)
  strides = [_execute.make_int(_i, "strides") for _i in strides]
  padding = _execute.make_str(padding, "padding")
  if out_type is None:
    out_type = _dtypes.qint32
  out_type = _execute.make_type(out_type, "out_type")
  if dilations is None:
    dilations = [1, 1, 1, 1]
  if not isinstance(dilations, (list, tuple)):
    raise TypeError(
        "Expected list for 'dilations' argument to "
        "'quantized_conv2d_with_bias_and_relu' Op, not %r." % dilations)
  dilations = [_execute.make_int(_i, "dilations") for _i in dilations]
  if padding_list is None:
    padding_list = []
  if not isinstance(padding_list, (list, tuple)):
    raise TypeError(
        "Expected list for 'padding_list' argument to "
        "'quantized_conv2d_with_bias_and_relu' Op, not %r." % padding_list)
  padding_list = [_execute.make_int(_i, "padding_list") for _i in padding_list]
  _attr_Tinput, (input,) = _execute.args_to_matching_eager([input], ctx, [_dtypes.qint8, _dtypes.quint8, _dtypes.qint32, _dtypes.qint16, _dtypes.quint16, ])
  _attr_Tfilter, (filter,) = _execute.args_to_matching_eager([filter], ctx, [_dtypes.qint8, _dtypes.quint8, _dtypes.qint32, _dtypes.qint16, _dtypes.quint16, ])
  bias = _ops.convert_to_tensor(bias, _dtypes.float32)
  min_input = _ops.convert_to_tensor(min_input, _dtypes.float32)
  max_input = _ops.convert_to_tensor(max_input, _dtypes.float32)
  min_filter = _ops.convert_to_tensor(min_filter, _dtypes.float32)
  max_filter = _ops.convert_to_tensor(max_filter, _dtypes.float32)
  _inputs_flat = [input, filter, bias, min_input, max_input, min_filter, max_filter]
  _attrs = ("Tinput", _attr_Tinput, "Tfilter", _attr_Tfilter, "out_type",
  out_type, "strides", strides, "padding", padding, "dilations", dilations,
  "padding_list", padding_list)
  _result = _execute.execute(b"QuantizedConv2DWithBiasAndRelu", 3,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "QuantizedConv2DWithBiasAndRelu", _inputs_flat, _attrs, _result)
  _result = _QuantizedConv2DWithBiasAndReluOutput._make(_result)
  return _result

_QuantizedConv2DWithBiasAndReluAndRequantizeOutput = collections.namedtuple(
    "QuantizedConv2DWithBiasAndReluAndRequantize",
    ["output", "min_output", "max_output"])


TV_QuantizedConv2DWithBiasAndReluAndRequantize_Tinput = TypeVar("TV_QuantizedConv2DWithBiasAndReluAndRequantize_Tinput", _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8)
TV_QuantizedConv2DWithBiasAndReluAndRequantize_Tfilter = TypeVar("TV_QuantizedConv2DWithBiasAndReluAndRequantize_Tfilter", _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8)
TV_QuantizedConv2DWithBiasAndReluAndRequantize_Tbias = TypeVar("TV_QuantizedConv2DWithBiasAndReluAndRequantize_Tbias", _atypes.Float32, _atypes.QInt32)
TV_QuantizedConv2DWithBiasAndReluAndRequantize_out_type = TypeVar("TV_QuantizedConv2DWithBiasAndReluAndRequantize_out_type", _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8)

def quantized_conv2d_with_bias_and_relu_and_requantize(input: Annotated[Any, TV_QuantizedConv2DWithBiasAndReluAndRequantize_Tinput], filter: Annotated[Any, TV_QuantizedConv2DWithBiasAndReluAndRequantize_Tfilter], bias: Annotated[Any, TV_QuantizedConv2DWithBiasAndReluAndRequantize_Tbias], min_input: Annotated[Any, _atypes.Float32], max_input: Annotated[Any, _atypes.Float32], min_filter: Annotated[Any, _atypes.Float32], max_filter: Annotated[Any, _atypes.Float32], min_freezed_output: Annotated[Any, _atypes.Float32], max_freezed_output: Annotated[Any, _atypes.Float32], strides, padding: str, out_type:TV_QuantizedConv2DWithBiasAndReluAndRequantize_out_type=_dtypes.quint8, dilations=[1, 1, 1, 1], padding_list=[], name=None):
  r"""TODO: add doc.

  Args:
    input: A `Tensor`. Must be one of the following types: `qint8`, `quint8`, `qint32`, `qint16`, `quint16`.
    filter: A `Tensor`. Must be one of the following types: `qint8`, `quint8`, `qint32`, `qint16`, `quint16`.
    bias: A `Tensor`. Must be one of the following types: `float32`, `qint32`.
    min_input: A `Tensor` of type `float32`.
    max_input: A `Tensor` of type `float32`.
    min_filter: A `Tensor` of type `float32`.
    max_filter: A `Tensor` of type `float32`.
    min_freezed_output: A `Tensor` of type `float32`.
    max_freezed_output: A `Tensor` of type `float32`.
    strides: A list of `ints`.
    padding: A `string` from: `"SAME", "VALID"`.
    out_type: An optional `tf.DType` from: `tf.qint8, tf.quint8, tf.qint32, tf.qint16, tf.quint16`. Defaults to `tf.quint8`.
    dilations: An optional list of `ints`. Defaults to `[1, 1, 1, 1]`.
    padding_list: An optional list of `ints`. Defaults to `[]`.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (output, min_output, max_output).

    output: A `Tensor` of type `out_type`.
    min_output: A `Tensor` of type `float32`.
    max_output: A `Tensor` of type `float32`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "QuantizedConv2DWithBiasAndReluAndRequantize", name, input,
        filter, bias, min_input, max_input, min_filter, max_filter,
        min_freezed_output, max_freezed_output, "out_type", out_type,
        "strides", strides, "padding", padding, "dilations", dilations,
        "padding_list", padding_list)
      _result = _QuantizedConv2DWithBiasAndReluAndRequantizeOutput._make(_result)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return quantized_conv2d_with_bias_and_relu_and_requantize_eager_fallback(
          input, filter, bias, min_input, max_input, min_filter, max_filter,
          min_freezed_output, max_freezed_output, out_type=out_type,
          strides=strides, padding=padding, dilations=dilations,
          padding_list=padding_list, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if not isinstance(strides, (list, tuple)):
    raise TypeError(
        "Expected list for 'strides' argument to "
        "'quantized_conv2d_with_bias_and_relu_and_requantize' Op, not %r." % strides)
  strides = [_execute.make_int(_i, "strides") for _i in strides]
  padding = _execute.make_str(padding, "padding")
  if out_type is None:
    out_type = _dtypes.quint8
  out_type = _execute.make_type(out_type, "out_type")
  if dilations is None:
    dilations = [1, 1, 1, 1]
  if not isinstance(dilations, (list, tuple)):
    raise TypeError(
        "Expected list for 'dilations' argument to "
        "'quantized_conv2d_with_bias_and_relu_and_requantize' Op, not %r." % dilations)
  dilations = [_execute.make_int(_i, "dilations") for _i in dilations]
  if padding_list is None:
    padding_list = []
  if not isinstance(padding_list, (list, tuple)):
    raise TypeError(
        "Expected list for 'padding_list' argument to "
        "'quantized_conv2d_with_bias_and_relu_and_requantize' Op, not %r." % padding_list)
  padding_list = [_execute.make_int(_i, "padding_list") for _i in padding_list]
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "QuantizedConv2DWithBiasAndReluAndRequantize", input=input,
                                                       filter=filter,
                                                       bias=bias,
                                                       min_input=min_input,
                                                       max_input=max_input,
                                                       min_filter=min_filter,
                                                       max_filter=max_filter,
                                                       min_freezed_output=min_freezed_output,
                                                       max_freezed_output=max_freezed_output,
                                                       strides=strides,
                                                       padding=padding,
                                                       out_type=out_type,
                                                       dilations=dilations,
                                                       padding_list=padding_list,
                                                       name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("Tinput", _op._get_attr_type("Tinput"), "Tfilter",
              _op._get_attr_type("Tfilter"), "Tbias",
              _op._get_attr_type("Tbias"), "out_type",
              _op._get_attr_type("out_type"), "strides",
              _op.get_attr("strides"), "padding", _op.get_attr("padding"),
              "dilations", _op.get_attr("dilations"), "padding_list",
              _op.get_attr("padding_list"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "QuantizedConv2DWithBiasAndReluAndRequantize", _inputs_flat, _attrs, _result)
  _result = _QuantizedConv2DWithBiasAndReluAndRequantizeOutput._make(_result)
  return _result

QuantizedConv2DWithBiasAndReluAndRequantize = tf_export("raw_ops.QuantizedConv2DWithBiasAndReluAndRequantize")(_ops.to_raw_op(quantized_conv2d_with_bias_and_relu_and_requantize))


def quantized_conv2d_with_bias_and_relu_and_requantize_eager_fallback(input: Annotated[Any, TV_QuantizedConv2DWithBiasAndReluAndRequantize_Tinput], filter: Annotated[Any, TV_QuantizedConv2DWithBiasAndReluAndRequantize_Tfilter], bias: Annotated[Any, TV_QuantizedConv2DWithBiasAndReluAndRequantize_Tbias], min_input: Annotated[Any, _atypes.Float32], max_input: Annotated[Any, _atypes.Float32], min_filter: Annotated[Any, _atypes.Float32], max_filter: Annotated[Any, _atypes.Float32], min_freezed_output: Annotated[Any, _atypes.Float32], max_freezed_output: Annotated[Any, _atypes.Float32], strides, padding: str, out_type: TV_QuantizedConv2DWithBiasAndReluAndRequantize_out_type, dilations, padding_list, name, ctx):
  if not isinstance(strides, (list, tuple)):
    raise TypeError(
        "Expected list for 'strides' argument to "
        "'quantized_conv2d_with_bias_and_relu_and_requantize' Op, not %r." % strides)
  strides = [_execute.make_int(_i, "strides") for _i in strides]
  padding = _execute.make_str(padding, "padding")
  if out_type is None:
    out_type = _dtypes.quint8
  out_type = _execute.make_type(out_type, "out_type")
  if dilations is None:
    dilations = [1, 1, 1, 1]
  if not isinstance(dilations, (list, tuple)):
    raise TypeError(
        "Expected list for 'dilations' argument to "
        "'quantized_conv2d_with_bias_and_relu_and_requantize' Op, not %r." % dilations)
  dilations = [_execute.make_int(_i, "dilations") for _i in dilations]
  if padding_list is None:
    padding_list = []
  if not isinstance(padding_list, (list, tuple)):
    raise TypeError(
        "Expected list for 'padding_list' argument to "
        "'quantized_conv2d_with_bias_and_relu_and_requantize' Op, not %r." % padding_list)
  padding_list = [_execute.make_int(_i, "padding_list") for _i in padding_list]
  _attr_Tinput, (input,) = _execute.args_to_matching_eager([input], ctx, [_dtypes.qint8, _dtypes.quint8, _dtypes.qint32, _dtypes.qint16, _dtypes.quint16, ])
  _attr_Tfilter, (filter,) = _execute.args_to_matching_eager([filter], ctx, [_dtypes.qint8, _dtypes.quint8, _dtypes.qint32, _dtypes.qint16, _dtypes.quint16, ])
  _attr_Tbias, (bias,) = _execute.args_to_matching_eager([bias], ctx, [_dtypes.float32, _dtypes.qint32, ])
  min_input = _ops.convert_to_tensor(min_input, _dtypes.float32)
  max_input = _ops.convert_to_tensor(max_input, _dtypes.float32)
  min_filter = _ops.convert_to_tensor(min_filter, _dtypes.float32)
  max_filter = _ops.convert_to_tensor(max_filter, _dtypes.float32)
  min_freezed_output = _ops.convert_to_tensor(min_freezed_output, _dtypes.float32)
  max_freezed_output = _ops.convert_to_tensor(max_freezed_output, _dtypes.float32)
  _inputs_flat = [input, filter, bias, min_input, max_input, min_filter, max_filter, min_freezed_output, max_freezed_output]
  _attrs = ("Tinput", _attr_Tinput, "Tfilter", _attr_Tfilter, "Tbias",
  _attr_Tbias, "out_type", out_type, "strides", strides, "padding", padding,
  "dilations", dilations, "padding_list", padding_list)
  _result = _execute.execute(b"QuantizedConv2DWithBiasAndReluAndRequantize",
                             3, inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "QuantizedConv2DWithBiasAndReluAndRequantize", _inputs_flat, _attrs, _result)
  _result = _QuantizedConv2DWithBiasAndReluAndRequantizeOutput._make(_result)
  return _result

_QuantizedConv2DWithBiasAndRequantizeOutput = collections.namedtuple(
    "QuantizedConv2DWithBiasAndRequantize",
    ["output", "min_output", "max_output"])


TV_QuantizedConv2DWithBiasAndRequantize_Tinput = TypeVar("TV_QuantizedConv2DWithBiasAndRequantize_Tinput", _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8)
TV_QuantizedConv2DWithBiasAndRequantize_Tfilter = TypeVar("TV_QuantizedConv2DWithBiasAndRequantize_Tfilter", _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8)
TV_QuantizedConv2DWithBiasAndRequantize_Tbias = TypeVar("TV_QuantizedConv2DWithBiasAndRequantize_Tbias", _atypes.Float32, _atypes.QInt32)
TV_QuantizedConv2DWithBiasAndRequantize_out_type = TypeVar("TV_QuantizedConv2DWithBiasAndRequantize_out_type", _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8)

def quantized_conv2d_with_bias_and_requantize(input: Annotated[Any, TV_QuantizedConv2DWithBiasAndRequantize_Tinput], filter: Annotated[Any, TV_QuantizedConv2DWithBiasAndRequantize_Tfilter], bias: Annotated[Any, TV_QuantizedConv2DWithBiasAndRequantize_Tbias], min_input: Annotated[Any, _atypes.Float32], max_input: Annotated[Any, _atypes.Float32], min_filter: Annotated[Any, _atypes.Float32], max_filter: Annotated[Any, _atypes.Float32], min_freezed_output: Annotated[Any, _atypes.Float32], max_freezed_output: Annotated[Any, _atypes.Float32], strides, padding: str, out_type:TV_QuantizedConv2DWithBiasAndRequantize_out_type=_dtypes.qint8, dilations=[1, 1, 1, 1], padding_list=[], name=None):
  r"""TODO: add doc.

  Args:
    input: A `Tensor`. Must be one of the following types: `qint8`, `quint8`, `qint32`, `qint16`, `quint16`.
    filter: A `Tensor`. Must be one of the following types: `qint8`, `quint8`, `qint32`, `qint16`, `quint16`.
    bias: A `Tensor`. Must be one of the following types: `float32`, `qint32`.
    min_input: A `Tensor` of type `float32`.
    max_input: A `Tensor` of type `float32`.
    min_filter: A `Tensor` of type `float32`.
    max_filter: A `Tensor` of type `float32`.
    min_freezed_output: A `Tensor` of type `float32`.
    max_freezed_output: A `Tensor` of type `float32`.
    strides: A list of `ints`.
    padding: A `string` from: `"SAME", "VALID"`.
    out_type: An optional `tf.DType` from: `tf.qint8, tf.quint8, tf.qint32, tf.qint16, tf.quint16`. Defaults to `tf.qint8`.
    dilations: An optional list of `ints`. Defaults to `[1, 1, 1, 1]`.
    padding_list: An optional list of `ints`. Defaults to `[]`.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (output, min_output, max_output).

    output: A `Tensor` of type `out_type`.
    min_output: A `Tensor` of type `float32`.
    max_output: A `Tensor` of type `float32`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "QuantizedConv2DWithBiasAndRequantize", name, input, filter,
        bias, min_input, max_input, min_filter, max_filter,
        min_freezed_output, max_freezed_output, "out_type", out_type,
        "strides", strides, "padding", padding, "dilations", dilations,
        "padding_list", padding_list)
      _result = _QuantizedConv2DWithBiasAndRequantizeOutput._make(_result)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return quantized_conv2d_with_bias_and_requantize_eager_fallback(
          input, filter, bias, min_input, max_input, min_filter, max_filter,
          min_freezed_output, max_freezed_output, out_type=out_type,
          strides=strides, padding=padding, dilations=dilations,
          padding_list=padding_list, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if not isinstance(strides, (list, tuple)):
    raise TypeError(
        "Expected list for 'strides' argument to "
        "'quantized_conv2d_with_bias_and_requantize' Op, not %r." % strides)
  strides = [_execute.make_int(_i, "strides") for _i in strides]
  padding = _execute.make_str(padding, "padding")
  if out_type is None:
    out_type = _dtypes.qint8
  out_type = _execute.make_type(out_type, "out_type")
  if dilations is None:
    dilations = [1, 1, 1, 1]
  if not isinstance(dilations, (list, tuple)):
    raise TypeError(
        "Expected list for 'dilations' argument to "
        "'quantized_conv2d_with_bias_and_requantize' Op, not %r." % dilations)
  dilations = [_execute.make_int(_i, "dilations") for _i in dilations]
  if padding_list is None:
    padding_list = []
  if not isinstance(padding_list, (list, tuple)):
    raise TypeError(
        "Expected list for 'padding_list' argument to "
        "'quantized_conv2d_with_bias_and_requantize' Op, not %r." % padding_list)
  padding_list = [_execute.make_int(_i, "padding_list") for _i in padding_list]
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "QuantizedConv2DWithBiasAndRequantize", input=input, filter=filter,
                                                bias=bias,
                                                min_input=min_input,
                                                max_input=max_input,
                                                min_filter=min_filter,
                                                max_filter=max_filter,
                                                min_freezed_output=min_freezed_output,
                                                max_freezed_output=max_freezed_output,
                                                strides=strides,
                                                padding=padding,
                                                out_type=out_type,
                                                dilations=dilations,
                                                padding_list=padding_list,
                                                name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("Tinput", _op._get_attr_type("Tinput"), "Tfilter",
              _op._get_attr_type("Tfilter"), "Tbias",
              _op._get_attr_type("Tbias"), "out_type",
              _op._get_attr_type("out_type"), "strides",
              _op.get_attr("strides"), "padding", _op.get_attr("padding"),
              "dilations", _op.get_attr("dilations"), "padding_list",
              _op.get_attr("padding_list"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "QuantizedConv2DWithBiasAndRequantize", _inputs_flat, _attrs, _result)
  _result = _QuantizedConv2DWithBiasAndRequantizeOutput._make(_result)
  return _result

QuantizedConv2DWithBiasAndRequantize = tf_export("raw_ops.QuantizedConv2DWithBiasAndRequantize")(_ops.to_raw_op(quantized_conv2d_with_bias_and_requantize))


def quantized_conv2d_with_bias_and_requantize_eager_fallback(input: Annotated[Any, TV_QuantizedConv2DWithBiasAndRequantize_Tinput], filter: Annotated[Any, TV_QuantizedConv2DWithBiasAndRequantize_Tfilter], bias: Annotated[Any, TV_QuantizedConv2DWithBiasAndRequantize_Tbias], min_input: Annotated[Any, _atypes.Float32], max_input: Annotated[Any, _atypes.Float32], min_filter: Annotated[Any, _atypes.Float32], max_filter: Annotated[Any, _atypes.Float32], min_freezed_output: Annotated[Any, _atypes.Float32], max_freezed_output: Annotated[Any, _atypes.Float32], strides, padding: str, out_type: TV_QuantizedConv2DWithBiasAndRequantize_out_type, dilations, padding_list, name, ctx):
  if not isinstance(strides, (list, tuple)):
    raise TypeError(
        "Expected list for 'strides' argument to "
        "'quantized_conv2d_with_bias_and_requantize' Op, not %r." % strides)
  strides = [_execute.make_int(_i, "strides") for _i in strides]
  padding = _execute.make_str(padding, "padding")
  if out_type is None:
    out_type = _dtypes.qint8
  out_type = _execute.make_type(out_type, "out_type")
  if dilations is None:
    dilations = [1, 1, 1, 1]
  if not isinstance(dilations, (list, tuple)):
    raise TypeError(
        "Expected list for 'dilations' argument to "
        "'quantized_conv2d_with_bias_and_requantize' Op, not %r." % dilations)
  dilations = [_execute.make_int(_i, "dilations") for _i in dilations]
  if padding_list is None:
    padding_list = []
  if not isinstance(padding_list, (list, tuple)):
    raise TypeError(
        "Expected list for 'padding_list' argument to "
        "'quantized_conv2d_with_bias_and_requantize' Op, not %r." % padding_list)
  padding_list = [_execute.make_int(_i, "padding_list") for _i in padding_list]
  _attr_Tinput, (input,) = _execute.args_to_matching_eager([input], ctx, [_dtypes.qint8, _dtypes.quint8, _dtypes.qint32, _dtypes.qint16, _dtypes.quint16, ])
  _attr_Tfilter, (filter,) = _execute.args_to_matching_eager([filter], ctx, [_dtypes.qint8, _dtypes.quint8, _dtypes.qint32, _dtypes.qint16, _dtypes.quint16, ])
  _attr_Tbias, (bias,) = _execute.args_to_matching_eager([bias], ctx, [_dtypes.float32, _dtypes.qint32, ])
  min_input = _ops.convert_to_tensor(min_input, _dtypes.float32)
  max_input = _ops.convert_to_tensor(max_input, _dtypes.float32)
  min_filter = _ops.convert_to_tensor(min_filter, _dtypes.float32)
  max_filter = _ops.convert_to_tensor(max_filter, _dtypes.float32)
  min_freezed_output = _ops.convert_to_tensor(min_freezed_output, _dtypes.float32)
  max_freezed_output = _ops.convert_to_tensor(max_freezed_output, _dtypes.float32)
  _inputs_flat = [input, filter, bias, min_input, max_input, min_filter, max_filter, min_freezed_output, max_freezed_output]
  _attrs = ("Tinput", _attr_Tinput, "Tfilter", _attr_Tfilter, "Tbias",
  _attr_Tbias, "out_type", out_type, "strides", strides, "padding", padding,
  "dilations", dilations, "padding_list", padding_list)
  _result = _execute.execute(b"QuantizedConv2DWithBiasAndRequantize", 3,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "QuantizedConv2DWithBiasAndRequantize", _inputs_flat, _attrs, _result)
  _result = _QuantizedConv2DWithBiasAndRequantizeOutput._make(_result)
  return _result

_QuantizedConv2DWithBiasSignedSumAndReluAndRequantizeOutput = collections.namedtuple(
    "QuantizedConv2DWithBiasSignedSumAndReluAndRequantize",
    ["output", "min_output", "max_output"])


TV_QuantizedConv2DWithBiasSignedSumAndReluAndRequantize_Tinput = TypeVar("TV_QuantizedConv2DWithBiasSignedSumAndReluAndRequantize_Tinput", _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8)
TV_QuantizedConv2DWithBiasSignedSumAndReluAndRequantize_Tfilter = TypeVar("TV_QuantizedConv2DWithBiasSignedSumAndReluAndRequantize_Tfilter", _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8)
TV_QuantizedConv2DWithBiasSignedSumAndReluAndRequantize_Tbias = TypeVar("TV_QuantizedConv2DWithBiasSignedSumAndReluAndRequantize_Tbias", _atypes.Float32, _atypes.QInt32)
TV_QuantizedConv2DWithBiasSignedSumAndReluAndRequantize_Tsummand = TypeVar("TV_QuantizedConv2DWithBiasSignedSumAndReluAndRequantize_Tsummand", _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8)
TV_QuantizedConv2DWithBiasSignedSumAndReluAndRequantize_out_type = TypeVar("TV_QuantizedConv2DWithBiasSignedSumAndReluAndRequantize_out_type", _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8)

def quantized_conv2d_with_bias_signed_sum_and_relu_and_requantize(input: Annotated[Any, TV_QuantizedConv2DWithBiasSignedSumAndReluAndRequantize_Tinput], filter: Annotated[Any, TV_QuantizedConv2DWithBiasSignedSumAndReluAndRequantize_Tfilter], bias: Annotated[Any, TV_QuantizedConv2DWithBiasSignedSumAndReluAndRequantize_Tbias], min_input: Annotated[Any, _atypes.Float32], max_input: Annotated[Any, _atypes.Float32], min_filter: Annotated[Any, _atypes.Float32], max_filter: Annotated[Any, _atypes.Float32], min_freezed_output: Annotated[Any, _atypes.Float32], max_freezed_output: Annotated[Any, _atypes.Float32], summand: Annotated[Any, TV_QuantizedConv2DWithBiasSignedSumAndReluAndRequantize_Tsummand], min_summand: Annotated[Any, _atypes.Float32], max_summand: Annotated[Any, _atypes.Float32], strides, padding: str, out_type:TV_QuantizedConv2DWithBiasSignedSumAndReluAndRequantize_out_type=_dtypes.quint8, dilations=[1, 1, 1, 1], padding_list=[], name=None):
  r"""TODO: add doc.

  Args:
    input: A `Tensor`. Must be one of the following types: `qint8`, `quint8`, `qint32`, `qint16`, `quint16`.
    filter: A `Tensor`. Must be one of the following types: `qint8`, `quint8`, `qint32`, `qint16`, `quint16`.
    bias: A `Tensor`. Must be one of the following types: `float32`, `qint32`.
    min_input: A `Tensor` of type `float32`.
    max_input: A `Tensor` of type `float32`.
    min_filter: A `Tensor` of type `float32`.
    max_filter: A `Tensor` of type `float32`.
    min_freezed_output: A `Tensor` of type `float32`.
    max_freezed_output: A `Tensor` of type `float32`.
    summand: A `Tensor`. Must be one of the following types: `qint8`, `quint8`, `qint32`, `qint16`, `quint16`.
    min_summand: A `Tensor` of type `float32`.
    max_summand: A `Tensor` of type `float32`.
    strides: A list of `ints`.
    padding: A `string` from: `"SAME", "VALID"`.
    out_type: An optional `tf.DType` from: `tf.qint8, tf.quint8, tf.qint32, tf.qint16, tf.quint16`. Defaults to `tf.quint8`.
    dilations: An optional list of `ints`. Defaults to `[1, 1, 1, 1]`.
    padding_list: An optional list of `ints`. Defaults to `[]`.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (output, min_output, max_output).

    output: A `Tensor` of type `out_type`.
    min_output: A `Tensor` of type `float32`.
    max_output: A `Tensor` of type `float32`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "QuantizedConv2DWithBiasSignedSumAndReluAndRequantize", name,
        input, filter, bias, min_input, max_input, min_filter, max_filter,
        min_freezed_output, max_freezed_output, summand, min_summand,
        max_summand, "out_type", out_type, "strides", strides, "padding",
        padding, "dilations", dilations, "padding_list", padding_list)
      _result = _QuantizedConv2DWithBiasSignedSumAndReluAndRequantizeOutput._make(_result)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return quantized_conv2d_with_bias_signed_sum_and_relu_and_requantize_eager_fallback(
          input, filter, bias, min_input, max_input, min_filter, max_filter,
          min_freezed_output, max_freezed_output, summand, min_summand,
          max_summand, out_type=out_type, strides=strides, padding=padding,
          dilations=dilations, padding_list=padding_list, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if not isinstance(strides, (list, tuple)):
    raise TypeError(
        "Expected list for 'strides' argument to "
        "'quantized_conv2d_with_bias_signed_sum_and_relu_and_requantize' Op, not %r." % strides)
  strides = [_execute.make_int(_i, "strides") for _i in strides]
  padding = _execute.make_str(padding, "padding")
  if out_type is None:
    out_type = _dtypes.quint8
  out_type = _execute.make_type(out_type, "out_type")
  if dilations is None:
    dilations = [1, 1, 1, 1]
  if not isinstance(dilations, (list, tuple)):
    raise TypeError(
        "Expected list for 'dilations' argument to "
        "'quantized_conv2d_with_bias_signed_sum_and_relu_and_requantize' Op, not %r." % dilations)
  dilations = [_execute.make_int(_i, "dilations") for _i in dilations]
  if padding_list is None:
    padding_list = []
  if not isinstance(padding_list, (list, tuple)):
    raise TypeError(
        "Expected list for 'padding_list' argument to "
        "'quantized_conv2d_with_bias_signed_sum_and_relu_and_requantize' Op, not %r." % padding_list)
  padding_list = [_execute.make_int(_i, "padding_list") for _i in padding_list]
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "QuantizedConv2DWithBiasSignedSumAndReluAndRequantize", input=input,
                                                                filter=filter,
                                                                bias=bias,
                                                                min_input=min_input,
                                                                max_input=max_input,
                                                                min_filter=min_filter,
                                                                max_filter=max_filter,
                                                                min_freezed_output=min_freezed_output,
                                                                max_freezed_output=max_freezed_output,
                                                                summand=summand,
                                                                min_summand=min_summand,
                                                                max_summand=max_summand,
                                                                strides=strides,
                                                                padding=padding,
                                                                out_type=out_type,
                                                                dilations=dilations,
                                                                padding_list=padding_list,
                                                                name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("Tinput", _op._get_attr_type("Tinput"), "Tfilter",
              _op._get_attr_type("Tfilter"), "Tbias",
              _op._get_attr_type("Tbias"), "Tsummand",
              _op._get_attr_type("Tsummand"), "out_type",
              _op._get_attr_type("out_type"), "strides",
              _op.get_attr("strides"), "padding", _op.get_attr("padding"),
              "dilations", _op.get_attr("dilations"), "padding_list",
              _op.get_attr("padding_list"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "QuantizedConv2DWithBiasSignedSumAndReluAndRequantize", _inputs_flat, _attrs, _result)
  _result = _QuantizedConv2DWithBiasSignedSumAndReluAndRequantizeOutput._make(_result)
  return _result

QuantizedConv2DWithBiasSignedSumAndReluAndRequantize = tf_export("raw_ops.QuantizedConv2DWithBiasSignedSumAndReluAndRequantize")(_ops.to_raw_op(quantized_conv2d_with_bias_signed_sum_and_relu_and_requantize))


def quantized_conv2d_with_bias_signed_sum_and_relu_and_requantize_eager_fallback(input: Annotated[Any, TV_QuantizedConv2DWithBiasSignedSumAndReluAndRequantize_Tinput], filter: Annotated[Any, TV_QuantizedConv2DWithBiasSignedSumAndReluAndRequantize_Tfilter], bias: Annotated[Any, TV_QuantizedConv2DWithBiasSignedSumAndReluAndRequantize_Tbias], min_input: Annotated[Any, _atypes.Float32], max_input: Annotated[Any, _atypes.Float32], min_filter: Annotated[Any, _atypes.Float32], max_filter: Annotated[Any, _atypes.Float32], min_freezed_output: Annotated[Any, _atypes.Float32], max_freezed_output: Annotated[Any, _atypes.Float32], summand: Annotated[Any, TV_QuantizedConv2DWithBiasSignedSumAndReluAndRequantize_Tsummand], min_summand: Annotated[Any, _atypes.Float32], max_summand: Annotated[Any, _atypes.Float32], strides, padding: str, out_type: TV_QuantizedConv2DWithBiasSignedSumAndReluAndRequantize_out_type, dilations, padding_list, name, ctx):
  if not isinstance(strides, (list, tuple)):
    raise TypeError(
        "Expected list for 'strides' argument to "
        "'quantized_conv2d_with_bias_signed_sum_and_relu_and_requantize' Op, not %r." % strides)
  strides = [_execute.make_int(_i, "strides") for _i in strides]
  padding = _execute.make_str(padding, "padding")
  if out_type is None:
    out_type = _dtypes.quint8
  out_type = _execute.make_type(out_type, "out_type")
  if dilations is None:
    dilations = [1, 1, 1, 1]
  if not isinstance(dilations, (list, tuple)):
    raise TypeError(
        "Expected list for 'dilations' argument to "
        "'quantized_conv2d_with_bias_signed_sum_and_relu_and_requantize' Op, not %r." % dilations)
  dilations = [_execute.make_int(_i, "dilations") for _i in dilations]
  if padding_list is None:
    padding_list = []
  if not isinstance(padding_list, (list, tuple)):
    raise TypeError(
        "Expected list for 'padding_list' argument to "
        "'quantized_conv2d_with_bias_signed_sum_and_relu_and_requantize' Op, not %r." % padding_list)
  padding_list = [_execute.make_int(_i, "padding_list") for _i in padding_list]
  _attr_Tinput, (input,) = _execute.args_to_matching_eager([input], ctx, [_dtypes.qint8, _dtypes.quint8, _dtypes.qint32, _dtypes.qint16, _dtypes.quint16, ])
  _attr_Tfilter, (filter,) = _execute.args_to_matching_eager([filter], ctx, [_dtypes.qint8, _dtypes.quint8, _dtypes.qint32, _dtypes.qint16, _dtypes.quint16, ])
  _attr_Tbias, (bias,) = _execute.args_to_matching_eager([bias], ctx, [_dtypes.float32, _dtypes.qint32, ])
  _attr_Tsummand, (summand,) = _execute.args_to_matching_eager([summand], ctx, [_dtypes.qint8, _dtypes.quint8, _dtypes.qint32, _dtypes.qint16, _dtypes.quint16, ])
  min_input = _ops.convert_to_tensor(min_input, _dtypes.float32)
  max_input = _ops.convert_to_tensor(max_input, _dtypes.float32)
  min_filter = _ops.convert_to_tensor(min_filter, _dtypes.float32)
  max_filter = _ops.convert_to_tensor(max_filter, _dtypes.float32)
  min_freezed_output = _ops.convert_to_tensor(min_freezed_output, _dtypes.float32)
  max_freezed_output = _ops.convert_to_tensor(max_freezed_output, _dtypes.float32)
  min_summand = _ops.convert_to_tensor(min_summand, _dtypes.float32)
  max_summand = _ops.convert_to_tensor(max_summand, _dtypes.float32)
  _inputs_flat = [input, filter, bias, min_input, max_input, min_filter, max_filter, min_freezed_output, max_freezed_output, summand, min_summand, max_summand]
  _attrs = ("Tinput", _attr_Tinput, "Tfilter", _attr_Tfilter, "Tbias",
  _attr_Tbias, "Tsummand", _attr_Tsummand, "out_type", out_type, "strides",
  strides, "padding", padding, "dilations", dilations, "padding_list",
  padding_list)
  _result = _execute.execute(b"QuantizedConv2DWithBiasSignedSumAndReluAndRequantize",
                             3, inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "QuantizedConv2DWithBiasSignedSumAndReluAndRequantize", _inputs_flat, _attrs, _result)
  _result = _QuantizedConv2DWithBiasSignedSumAndReluAndRequantizeOutput._make(_result)
  return _result

_QuantizedConv2DWithBiasSumAndReluOutput = collections.namedtuple(
    "QuantizedConv2DWithBiasSumAndRelu",
    ["output", "min_output", "max_output"])


TV_QuantizedConv2DWithBiasSumAndRelu_Tinput = TypeVar("TV_QuantizedConv2DWithBiasSumAndRelu_Tinput", _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8)
TV_QuantizedConv2DWithBiasSumAndRelu_Tfilter = TypeVar("TV_QuantizedConv2DWithBiasSumAndRelu_Tfilter", _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8)
TV_QuantizedConv2DWithBiasSumAndRelu_out_type = TypeVar("TV_QuantizedConv2DWithBiasSumAndRelu_out_type", _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8)

def quantized_conv2d_with_bias_sum_and_relu(input: Annotated[Any, TV_QuantizedConv2DWithBiasSumAndRelu_Tinput], filter: Annotated[Any, TV_QuantizedConv2DWithBiasSumAndRelu_Tfilter], bias: Annotated[Any, _atypes.Float32], min_input: Annotated[Any, _atypes.Float32], max_input: Annotated[Any, _atypes.Float32], min_filter: Annotated[Any, _atypes.Float32], max_filter: Annotated[Any, _atypes.Float32], summand: Annotated[Any, _atypes.Float32], strides, padding: str, out_type:TV_QuantizedConv2DWithBiasSumAndRelu_out_type=_dtypes.qint32, dilations=[1, 1, 1, 1], padding_list=[], name=None):
  r"""TODO: add doc.

  Args:
    input: A `Tensor`. Must be one of the following types: `qint8`, `quint8`, `qint32`, `qint16`, `quint16`.
    filter: A `Tensor`. Must be one of the following types: `qint8`, `quint8`, `qint32`, `qint16`, `quint16`.
    bias: A `Tensor` of type `float32`.
    min_input: A `Tensor` of type `float32`.
    max_input: A `Tensor` of type `float32`.
    min_filter: A `Tensor` of type `float32`.
    max_filter: A `Tensor` of type `float32`.
    summand: A `Tensor` of type `float32`.
    strides: A list of `ints`.
    padding: A `string` from: `"SAME", "VALID"`.
    out_type: An optional `tf.DType` from: `tf.qint8, tf.quint8, tf.qint32, tf.qint16, tf.quint16`. Defaults to `tf.qint32`.
    dilations: An optional list of `ints`. Defaults to `[1, 1, 1, 1]`.
    padding_list: An optional list of `ints`. Defaults to `[]`.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (output, min_output, max_output).

    output: A `Tensor` of type `out_type`.
    min_output: A `Tensor` of type `float32`.
    max_output: A `Tensor` of type `float32`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "QuantizedConv2DWithBiasSumAndRelu", name, input, filter, bias,
        min_input, max_input, min_filter, max_filter, summand, "out_type",
        out_type, "strides", strides, "padding", padding, "dilations",
        dilations, "padding_list", padding_list)
      _result = _QuantizedConv2DWithBiasSumAndReluOutput._make(_result)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return quantized_conv2d_with_bias_sum_and_relu_eager_fallback(
          input, filter, bias, min_input, max_input, min_filter, max_filter,
          summand, out_type=out_type, strides=strides, padding=padding,
          dilations=dilations, padding_list=padding_list, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if not isinstance(strides, (list, tuple)):
    raise TypeError(
        "Expected list for 'strides' argument to "
        "'quantized_conv2d_with_bias_sum_and_relu' Op, not %r." % strides)
  strides = [_execute.make_int(_i, "strides") for _i in strides]
  padding = _execute.make_str(padding, "padding")
  if out_type is None:
    out_type = _dtypes.qint32
  out_type = _execute.make_type(out_type, "out_type")
  if dilations is None:
    dilations = [1, 1, 1, 1]
  if not isinstance(dilations, (list, tuple)):
    raise TypeError(
        "Expected list for 'dilations' argument to "
        "'quantized_conv2d_with_bias_sum_and_relu' Op, not %r." % dilations)
  dilations = [_execute.make_int(_i, "dilations") for _i in dilations]
  if padding_list is None:
    padding_list = []
  if not isinstance(padding_list, (list, tuple)):
    raise TypeError(
        "Expected list for 'padding_list' argument to "
        "'quantized_conv2d_with_bias_sum_and_relu' Op, not %r." % padding_list)
  padding_list = [_execute.make_int(_i, "padding_list") for _i in padding_list]
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "QuantizedConv2DWithBiasSumAndRelu", input=input, filter=filter,
                                             bias=bias, min_input=min_input,
                                             max_input=max_input,
                                             min_filter=min_filter,
                                             max_filter=max_filter,
                                             summand=summand, strides=strides,
                                             padding=padding,
                                             out_type=out_type,
                                             dilations=dilations,
                                             padding_list=padding_list,
                                             name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("Tinput", _op._get_attr_type("Tinput"), "Tfilter",
              _op._get_attr_type("Tfilter"), "out_type",
              _op._get_attr_type("out_type"), "strides",
              _op.get_attr("strides"), "padding", _op.get_attr("padding"),
              "dilations", _op.get_attr("dilations"), "padding_list",
              _op.get_attr("padding_list"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "QuantizedConv2DWithBiasSumAndRelu", _inputs_flat, _attrs, _result)
  _result = _QuantizedConv2DWithBiasSumAndReluOutput._make(_result)
  return _result

QuantizedConv2DWithBiasSumAndRelu = tf_export("raw_ops.QuantizedConv2DWithBiasSumAndRelu")(_ops.to_raw_op(quantized_conv2d_with_bias_sum_and_relu))


def quantized_conv2d_with_bias_sum_and_relu_eager_fallback(input: Annotated[Any, TV_QuantizedConv2DWithBiasSumAndRelu_Tinput], filter: Annotated[Any, TV_QuantizedConv2DWithBiasSumAndRelu_Tfilter], bias: Annotated[Any, _atypes.Float32], min_input: Annotated[Any, _atypes.Float32], max_input: Annotated[Any, _atypes.Float32], min_filter: Annotated[Any, _atypes.Float32], max_filter: Annotated[Any, _atypes.Float32], summand: Annotated[Any, _atypes.Float32], strides, padding: str, out_type: TV_QuantizedConv2DWithBiasSumAndRelu_out_type, dilations, padding_list, name, ctx):
  if not isinstance(strides, (list, tuple)):
    raise TypeError(
        "Expected list for 'strides' argument to "
        "'quantized_conv2d_with_bias_sum_and_relu' Op, not %r." % strides)
  strides = [_execute.make_int(_i, "strides") for _i in strides]
  padding = _execute.make_str(padding, "padding")
  if out_type is None:
    out_type = _dtypes.qint32
  out_type = _execute.make_type(out_type, "out_type")
  if dilations is None:
    dilations = [1, 1, 1, 1]
  if not isinstance(dilations, (list, tuple)):
    raise TypeError(
        "Expected list for 'dilations' argument to "
        "'quantized_conv2d_with_bias_sum_and_relu' Op, not %r." % dilations)
  dilations = [_execute.make_int(_i, "dilations") for _i in dilations]
  if padding_list is None:
    padding_list = []
  if not isinstance(padding_list, (list, tuple)):
    raise TypeError(
        "Expected list for 'padding_list' argument to "
        "'quantized_conv2d_with_bias_sum_and_relu' Op, not %r." % padding_list)
  padding_list = [_execute.make_int(_i, "padding_list") for _i in padding_list]
  _attr_Tinput, (input,) = _execute.args_to_matching_eager([input], ctx, [_dtypes.qint8, _dtypes.quint8, _dtypes.qint32, _dtypes.qint16, _dtypes.quint16, ])
  _attr_Tfilter, (filter,) = _execute.args_to_matching_eager([filter], ctx, [_dtypes.qint8, _dtypes.quint8, _dtypes.qint32, _dtypes.qint16, _dtypes.quint16, ])
  bias = _ops.convert_to_tensor(bias, _dtypes.float32)
  min_input = _ops.convert_to_tensor(min_input, _dtypes.float32)
  max_input = _ops.convert_to_tensor(max_input, _dtypes.float32)
  min_filter = _ops.convert_to_tensor(min_filter, _dtypes.float32)
  max_filter = _ops.convert_to_tensor(max_filter, _dtypes.float32)
  summand = _ops.convert_to_tensor(summand, _dtypes.float32)
  _inputs_flat = [input, filter, bias, min_input, max_input, min_filter, max_filter, summand]
  _attrs = ("Tinput", _attr_Tinput, "Tfilter", _attr_Tfilter, "out_type",
  out_type, "strides", strides, "padding", padding, "dilations", dilations,
  "padding_list", padding_list)
  _result = _execute.execute(b"QuantizedConv2DWithBiasSumAndRelu", 3,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "QuantizedConv2DWithBiasSumAndRelu", _inputs_flat, _attrs, _result)
  _result = _QuantizedConv2DWithBiasSumAndReluOutput._make(_result)
  return _result

_QuantizedConv2DWithBiasSumAndReluAndRequantizeOutput = collections.namedtuple(
    "QuantizedConv2DWithBiasSumAndReluAndRequantize",
    ["output", "min_output", "max_output"])


TV_QuantizedConv2DWithBiasSumAndReluAndRequantize_Tinput = TypeVar("TV_QuantizedConv2DWithBiasSumAndReluAndRequantize_Tinput", _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8)
TV_QuantizedConv2DWithBiasSumAndReluAndRequantize_Tfilter = TypeVar("TV_QuantizedConv2DWithBiasSumAndReluAndRequantize_Tfilter", _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8)
TV_QuantizedConv2DWithBiasSumAndReluAndRequantize_Tbias = TypeVar("TV_QuantizedConv2DWithBiasSumAndReluAndRequantize_Tbias", _atypes.Float32, _atypes.QInt32)
TV_QuantizedConv2DWithBiasSumAndReluAndRequantize_Tsummand = TypeVar("TV_QuantizedConv2DWithBiasSumAndReluAndRequantize_Tsummand", _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8)
TV_QuantizedConv2DWithBiasSumAndReluAndRequantize_out_type = TypeVar("TV_QuantizedConv2DWithBiasSumAndReluAndRequantize_out_type", _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8)

def quantized_conv2d_with_bias_sum_and_relu_and_requantize(input: Annotated[Any, TV_QuantizedConv2DWithBiasSumAndReluAndRequantize_Tinput], filter: Annotated[Any, TV_QuantizedConv2DWithBiasSumAndReluAndRequantize_Tfilter], bias: Annotated[Any, TV_QuantizedConv2DWithBiasSumAndReluAndRequantize_Tbias], min_input: Annotated[Any, _atypes.Float32], max_input: Annotated[Any, _atypes.Float32], min_filter: Annotated[Any, _atypes.Float32], max_filter: Annotated[Any, _atypes.Float32], min_freezed_output: Annotated[Any, _atypes.Float32], max_freezed_output: Annotated[Any, _atypes.Float32], summand: Annotated[Any, TV_QuantizedConv2DWithBiasSumAndReluAndRequantize_Tsummand], min_summand: Annotated[Any, _atypes.Float32], max_summand: Annotated[Any, _atypes.Float32], strides, padding: str, out_type:TV_QuantizedConv2DWithBiasSumAndReluAndRequantize_out_type=_dtypes.quint8, dilations=[1, 1, 1, 1], padding_list=[], name=None):
  r"""TODO: add doc.

  Args:
    input: A `Tensor`. Must be one of the following types: `qint8`, `quint8`, `qint32`, `qint16`, `quint16`.
    filter: A `Tensor`. Must be one of the following types: `qint8`, `quint8`, `qint32`, `qint16`, `quint16`.
    bias: A `Tensor`. Must be one of the following types: `float32`, `qint32`.
    min_input: A `Tensor` of type `float32`.
    max_input: A `Tensor` of type `float32`.
    min_filter: A `Tensor` of type `float32`.
    max_filter: A `Tensor` of type `float32`.
    min_freezed_output: A `Tensor` of type `float32`.
    max_freezed_output: A `Tensor` of type `float32`.
    summand: A `Tensor`. Must be one of the following types: `qint8`, `quint8`, `qint32`, `qint16`, `quint16`.
    min_summand: A `Tensor` of type `float32`.
    max_summand: A `Tensor` of type `float32`.
    strides: A list of `ints`.
    padding: A `string` from: `"SAME", "VALID"`.
    out_type: An optional `tf.DType` from: `tf.qint8, tf.quint8, tf.qint32, tf.qint16, tf.quint16`. Defaults to `tf.quint8`.
    dilations: An optional list of `ints`. Defaults to `[1, 1, 1, 1]`.
    padding_list: An optional list of `ints`. Defaults to `[]`.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (output, min_output, max_output).

    output: A `Tensor` of type `out_type`.
    min_output: A `Tensor` of type `float32`.
    max_output: A `Tensor` of type `float32`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "QuantizedConv2DWithBiasSumAndReluAndRequantize", name, input,
        filter, bias, min_input, max_input, min_filter, max_filter,
        min_freezed_output, max_freezed_output, summand, min_summand,
        max_summand, "out_type", out_type, "strides", strides, "padding",
        padding, "dilations", dilations, "padding_list", padding_list)
      _result = _QuantizedConv2DWithBiasSumAndReluAndRequantizeOutput._make(_result)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return quantized_conv2d_with_bias_sum_and_relu_and_requantize_eager_fallback(
          input, filter, bias, min_input, max_input, min_filter, max_filter,
          min_freezed_output, max_freezed_output, summand, min_summand,
          max_summand, out_type=out_type, strides=strides, padding=padding,
          dilations=dilations, padding_list=padding_list, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if not isinstance(strides, (list, tuple)):
    raise TypeError(
        "Expected list for 'strides' argument to "
        "'quantized_conv2d_with_bias_sum_and_relu_and_requantize' Op, not %r." % strides)
  strides = [_execute.make_int(_i, "strides") for _i in strides]
  padding = _execute.make_str(padding, "padding")
  if out_type is None:
    out_type = _dtypes.quint8
  out_type = _execute.make_type(out_type, "out_type")
  if dilations is None:
    dilations = [1, 1, 1, 1]
  if not isinstance(dilations, (list, tuple)):
    raise TypeError(
        "Expected list for 'dilations' argument to "
        "'quantized_conv2d_with_bias_sum_and_relu_and_requantize' Op, not %r." % dilations)
  dilations = [_execute.make_int(_i, "dilations") for _i in dilations]
  if padding_list is None:
    padding_list = []
  if not isinstance(padding_list, (list, tuple)):
    raise TypeError(
        "Expected list for 'padding_list' argument to "
        "'quantized_conv2d_with_bias_sum_and_relu_and_requantize' Op, not %r." % padding_list)
  padding_list = [_execute.make_int(_i, "padding_list") for _i in padding_list]
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "QuantizedConv2DWithBiasSumAndReluAndRequantize", input=input,
                                                          filter=filter,
                                                          bias=bias,
                                                          min_input=min_input,
                                                          max_input=max_input,
                                                          min_filter=min_filter,
                                                          max_filter=max_filter,
                                                          min_freezed_output=min_freezed_output,
                                                          max_freezed_output=max_freezed_output,
                                                          summand=summand,
                                                          min_summand=min_summand,
                                                          max_summand=max_summand,
                                                          strides=strides,
                                                          padding=padding,
                                                          out_type=out_type,
                                                          dilations=dilations,
                                                          padding_list=padding_list,
                                                          name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("Tinput", _op._get_attr_type("Tinput"), "Tfilter",
              _op._get_attr_type("Tfilter"), "Tbias",
              _op._get_attr_type("Tbias"), "Tsummand",
              _op._get_attr_type("Tsummand"), "out_type",
              _op._get_attr_type("out_type"), "strides",
              _op.get_attr("strides"), "padding", _op.get_attr("padding"),
              "dilations", _op.get_attr("dilations"), "padding_list",
              _op.get_attr("padding_list"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "QuantizedConv2DWithBiasSumAndReluAndRequantize", _inputs_flat, _attrs, _result)
  _result = _QuantizedConv2DWithBiasSumAndReluAndRequantizeOutput._make(_result)
  return _result

QuantizedConv2DWithBiasSumAndReluAndRequantize = tf_export("raw_ops.QuantizedConv2DWithBiasSumAndReluAndRequantize")(_ops.to_raw_op(quantized_conv2d_with_bias_sum_and_relu_and_requantize))


def quantized_conv2d_with_bias_sum_and_relu_and_requantize_eager_fallback(input: Annotated[Any, TV_QuantizedConv2DWithBiasSumAndReluAndRequantize_Tinput], filter: Annotated[Any, TV_QuantizedConv2DWithBiasSumAndReluAndRequantize_Tfilter], bias: Annotated[Any, TV_QuantizedConv2DWithBiasSumAndReluAndRequantize_Tbias], min_input: Annotated[Any, _atypes.Float32], max_input: Annotated[Any, _atypes.Float32], min_filter: Annotated[Any, _atypes.Float32], max_filter: Annotated[Any, _atypes.Float32], min_freezed_output: Annotated[Any, _atypes.Float32], max_freezed_output: Annotated[Any, _atypes.Float32], summand: Annotated[Any, TV_QuantizedConv2DWithBiasSumAndReluAndRequantize_Tsummand], min_summand: Annotated[Any, _atypes.Float32], max_summand: Annotated[Any, _atypes.Float32], strides, padding: str, out_type: TV_QuantizedConv2DWithBiasSumAndReluAndRequantize_out_type, dilations, padding_list, name, ctx):
  if not isinstance(strides, (list, tuple)):
    raise TypeError(
        "Expected list for 'strides' argument to "
        "'quantized_conv2d_with_bias_sum_and_relu_and_requantize' Op, not %r." % strides)
  strides = [_execute.make_int(_i, "strides") for _i in strides]
  padding = _execute.make_str(padding, "padding")
  if out_type is None:
    out_type = _dtypes.quint8
  out_type = _execute.make_type(out_type, "out_type")
  if dilations is None:
    dilations = [1, 1, 1, 1]
  if not isinstance(dilations, (list, tuple)):
    raise TypeError(
        "Expected list for 'dilations' argument to "
        "'quantized_conv2d_with_bias_sum_and_relu_and_requantize' Op, not %r." % dilations)
  dilations = [_execute.make_int(_i, "dilations") for _i in dilations]
  if padding_list is None:
    padding_list = []
  if not isinstance(padding_list, (list, tuple)):
    raise TypeError(
        "Expected list for 'padding_list' argument to "
        "'quantized_conv2d_with_bias_sum_and_relu_and_requantize' Op, not %r." % padding_list)
  padding_list = [_execute.make_int(_i, "padding_list") for _i in padding_list]
  _attr_Tinput, (input,) = _execute.args_to_matching_eager([input], ctx, [_dtypes.qint8, _dtypes.quint8, _dtypes.qint32, _dtypes.qint16, _dtypes.quint16, ])
  _attr_Tfilter, (filter,) = _execute.args_to_matching_eager([filter], ctx, [_dtypes.qint8, _dtypes.quint8, _dtypes.qint32, _dtypes.qint16, _dtypes.quint16, ])
  _attr_Tbias, (bias,) = _execute.args_to_matching_eager([bias], ctx, [_dtypes.float32, _dtypes.qint32, ])
  _attr_Tsummand, (summand,) = _execute.args_to_matching_eager([summand], ctx, [_dtypes.qint8, _dtypes.quint8, _dtypes.qint32, _dtypes.qint16, _dtypes.quint16, ])
  min_input = _ops.convert_to_tensor(min_input, _dtypes.float32)
  max_input = _ops.convert_to_tensor(max_input, _dtypes.float32)
  min_filter = _ops.convert_to_tensor(min_filter, _dtypes.float32)
  max_filter = _ops.convert_to_tensor(max_filter, _dtypes.float32)
  min_freezed_output = _ops.convert_to_tensor(min_freezed_output, _dtypes.float32)
  max_freezed_output = _ops.convert_to_tensor(max_freezed_output, _dtypes.float32)
  min_summand = _ops.convert_to_tensor(min_summand, _dtypes.float32)
  max_summand = _ops.convert_to_tensor(max_summand, _dtypes.float32)
  _inputs_flat = [input, filter, bias, min_input, max_input, min_filter, max_filter, min_freezed_output, max_freezed_output, summand, min_summand, max_summand]
  _attrs = ("Tinput", _attr_Tinput, "Tfilter", _attr_Tfilter, "Tbias",
  _attr_Tbias, "Tsummand", _attr_Tsummand, "out_type", out_type, "strides",
  strides, "padding", padding, "dilations", dilations, "padding_list",
  padding_list)
  _result = _execute.execute(b"QuantizedConv2DWithBiasSumAndReluAndRequantize",
                             3, inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "QuantizedConv2DWithBiasSumAndReluAndRequantize", _inputs_flat, _attrs, _result)
  _result = _QuantizedConv2DWithBiasSumAndReluAndRequantizeOutput._make(_result)
  return _result

_QuantizedDepthwiseConv2DOutput = collections.namedtuple(
    "QuantizedDepthwiseConv2D",
    ["output", "min_output", "max_output"])


TV_QuantizedDepthwiseConv2D_Tinput = TypeVar("TV_QuantizedDepthwiseConv2D_Tinput", _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8)
TV_QuantizedDepthwiseConv2D_Tfilter = TypeVar("TV_QuantizedDepthwiseConv2D_Tfilter", _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8)
TV_QuantizedDepthwiseConv2D_out_type = TypeVar("TV_QuantizedDepthwiseConv2D_out_type", _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8)

def quantized_depthwise_conv2d(input: Annotated[Any, TV_QuantizedDepthwiseConv2D_Tinput], filter: Annotated[Any, TV_QuantizedDepthwiseConv2D_Tfilter], min_input: Annotated[Any, _atypes.Float32], max_input: Annotated[Any, _atypes.Float32], min_filter: Annotated[Any, _atypes.Float32], max_filter: Annotated[Any, _atypes.Float32], strides, padding: str, out_type:TV_QuantizedDepthwiseConv2D_out_type=_dtypes.qint32, dilations=[1, 1, 1, 1], name=None):
  r"""Computes quantized depthwise Conv2D.

  Args:
    input: A `Tensor`. Must be one of the following types: `qint8`, `quint8`, `qint32`, `qint16`, `quint16`.
      The original input tensor.
    filter: A `Tensor`. Must be one of the following types: `qint8`, `quint8`, `qint32`, `qint16`, `quint16`.
      The original filter tensor.
    min_input: A `Tensor` of type `float32`.
      The float value that the minimum quantized input value represents.
    max_input: A `Tensor` of type `float32`.
      The float value that the maximum quantized input value represents.
    min_filter: A `Tensor` of type `float32`.
      The float value that the minimum quantized filter value represents.
    max_filter: A `Tensor` of type `float32`.
      The float value that the maximum quantized filter value represents.
    strides: A list of `ints`. List of stride values.
    padding: A `string` from: `"SAME", "VALID"`.
    out_type: An optional `tf.DType` from: `tf.qint8, tf.quint8, tf.qint32, tf.qint16, tf.quint16`. Defaults to `tf.qint32`.
      The type of the output.
    dilations: An optional list of `ints`. Defaults to `[1, 1, 1, 1]`.
      List of dilation values.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (output, min_output, max_output).

    output: A `Tensor` of type `out_type`.
    min_output: A `Tensor` of type `float32`.
    max_output: A `Tensor` of type `float32`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "QuantizedDepthwiseConv2D", name, input, filter, min_input,
        max_input, min_filter, max_filter, "out_type", out_type, "strides",
        strides, "padding", padding, "dilations", dilations)
      _result = _QuantizedDepthwiseConv2DOutput._make(_result)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return quantized_depthwise_conv2d_eager_fallback(
          input, filter, min_input, max_input, min_filter, max_filter,
          out_type=out_type, strides=strides, padding=padding,
          dilations=dilations, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if not isinstance(strides, (list, tuple)):
    raise TypeError(
        "Expected list for 'strides' argument to "
        "'quantized_depthwise_conv2d' Op, not %r." % strides)
  strides = [_execute.make_int(_i, "strides") for _i in strides]
  padding = _execute.make_str(padding, "padding")
  if out_type is None:
    out_type = _dtypes.qint32
  out_type = _execute.make_type(out_type, "out_type")
  if dilations is None:
    dilations = [1, 1, 1, 1]
  if not isinstance(dilations, (list, tuple)):
    raise TypeError(
        "Expected list for 'dilations' argument to "
        "'quantized_depthwise_conv2d' Op, not %r." % dilations)
  dilations = [_execute.make_int(_i, "dilations") for _i in dilations]
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "QuantizedDepthwiseConv2D", input=input, filter=filter,
                                    min_input=min_input, max_input=max_input,
                                    min_filter=min_filter,
                                    max_filter=max_filter, strides=strides,
                                    padding=padding, out_type=out_type,
                                    dilations=dilations, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("Tinput", _op._get_attr_type("Tinput"), "Tfilter",
              _op._get_attr_type("Tfilter"), "out_type",
              _op._get_attr_type("out_type"), "strides",
              _op.get_attr("strides"), "padding", _op.get_attr("padding"),
              "dilations", _op.get_attr("dilations"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "QuantizedDepthwiseConv2D", _inputs_flat, _attrs, _result)
  _result = _QuantizedDepthwiseConv2DOutput._make(_result)
  return _result

QuantizedDepthwiseConv2D = tf_export("raw_ops.QuantizedDepthwiseConv2D")(_ops.to_raw_op(quantized_depthwise_conv2d))


def quantized_depthwise_conv2d_eager_fallback(input: Annotated[Any, TV_QuantizedDepthwiseConv2D_Tinput], filter: Annotated[Any, TV_QuantizedDepthwiseConv2D_Tfilter], min_input: Annotated[Any, _atypes.Float32], max_input: Annotated[Any, _atypes.Float32], min_filter: Annotated[Any, _atypes.Float32], max_filter: Annotated[Any, _atypes.Float32], strides, padding: str, out_type: TV_QuantizedDepthwiseConv2D_out_type, dilations, name, ctx):
  if not isinstance(strides, (list, tuple)):
    raise TypeError(
        "Expected list for 'strides' argument to "
        "'quantized_depthwise_conv2d' Op, not %r." % strides)
  strides = [_execute.make_int(_i, "strides") for _i in strides]
  padding = _execute.make_str(padding, "padding")
  if out_type is None:
    out_type = _dtypes.qint32
  out_type = _execute.make_type(out_type, "out_type")
  if dilations is None:
    dilations = [1, 1, 1, 1]
  if not isinstance(dilations, (list, tuple)):
    raise TypeError(
        "Expected list for 'dilations' argument to "
        "'quantized_depthwise_conv2d' Op, not %r." % dilations)
  dilations = [_execute.make_int(_i, "dilations") for _i in dilations]
  _attr_Tinput, (input,) = _execute.args_to_matching_eager([input], ctx, [_dtypes.qint8, _dtypes.quint8, _dtypes.qint32, _dtypes.qint16, _dtypes.quint16, ])
  _attr_Tfilter, (filter,) = _execute.args_to_matching_eager([filter], ctx, [_dtypes.qint8, _dtypes.quint8, _dtypes.qint32, _dtypes.qint16, _dtypes.quint16, ])
  min_input = _ops.convert_to_tensor(min_input, _dtypes.float32)
  max_input = _ops.convert_to_tensor(max_input, _dtypes.float32)
  min_filter = _ops.convert_to_tensor(min_filter, _dtypes.float32)
  max_filter = _ops.convert_to_tensor(max_filter, _dtypes.float32)
  _inputs_flat = [input, filter, min_input, max_input, min_filter, max_filter]
  _attrs = ("Tinput", _attr_Tinput, "Tfilter", _attr_Tfilter, "out_type",
  out_type, "strides", strides, "padding", padding, "dilations", dilations)
  _result = _execute.execute(b"QuantizedDepthwiseConv2D", 3,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "QuantizedDepthwiseConv2D", _inputs_flat, _attrs, _result)
  _result = _QuantizedDepthwiseConv2DOutput._make(_result)
  return _result

_QuantizedDepthwiseConv2DWithBiasOutput = collections.namedtuple(
    "QuantizedDepthwiseConv2DWithBias",
    ["output", "min_output", "max_output"])


TV_QuantizedDepthwiseConv2DWithBias_Tinput = TypeVar("TV_QuantizedDepthwiseConv2DWithBias_Tinput", _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8)
TV_QuantizedDepthwiseConv2DWithBias_Tfilter = TypeVar("TV_QuantizedDepthwiseConv2DWithBias_Tfilter", _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8)
TV_QuantizedDepthwiseConv2DWithBias_out_type = TypeVar("TV_QuantizedDepthwiseConv2DWithBias_out_type", _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8)

def quantized_depthwise_conv2d_with_bias(input: Annotated[Any, TV_QuantizedDepthwiseConv2DWithBias_Tinput], filter: Annotated[Any, TV_QuantizedDepthwiseConv2DWithBias_Tfilter], bias: Annotated[Any, _atypes.Float32], min_input: Annotated[Any, _atypes.Float32], max_input: Annotated[Any, _atypes.Float32], min_filter: Annotated[Any, _atypes.Float32], max_filter: Annotated[Any, _atypes.Float32], strides, padding: str, out_type:TV_QuantizedDepthwiseConv2DWithBias_out_type=_dtypes.qint32, dilations=[1, 1, 1, 1], name=None):
  r"""Computes quantized depthwise Conv2D with Bias.

  Args:
    input: A `Tensor`. Must be one of the following types: `qint8`, `quint8`, `qint32`, `qint16`, `quint16`.
      The original input tensor.
    filter: A `Tensor`. Must be one of the following types: `qint8`, `quint8`, `qint32`, `qint16`, `quint16`.
      The original filter tensor.
    bias: A `Tensor` of type `float32`. The original bias tensor.
    min_input: A `Tensor` of type `float32`.
      The float value that the minimum quantized input value represents.
    max_input: A `Tensor` of type `float32`.
      The float value that the maximum quantized input value represents.
    min_filter: A `Tensor` of type `float32`.
      The float value that the minimum quantized filter value represents.
    max_filter: A `Tensor` of type `float32`.
      The float value that the maximum quantized filter value represents.
    strides: A list of `ints`. List of stride values.
    padding: A `string` from: `"SAME", "VALID"`.
    out_type: An optional `tf.DType` from: `tf.qint8, tf.quint8, tf.qint32, tf.qint16, tf.quint16`. Defaults to `tf.qint32`.
      The type of the output.
    dilations: An optional list of `ints`. Defaults to `[1, 1, 1, 1]`.
      List of dilation values.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (output, min_output, max_output).

    output: A `Tensor` of type `out_type`.
    min_output: A `Tensor` of type `float32`.
    max_output: A `Tensor` of type `float32`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "QuantizedDepthwiseConv2DWithBias", name, input, filter, bias,
        min_input, max_input, min_filter, max_filter, "out_type", out_type,
        "strides", strides, "padding", padding, "dilations", dilations)
      _result = _QuantizedDepthwiseConv2DWithBiasOutput._make(_result)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return quantized_depthwise_conv2d_with_bias_eager_fallback(
          input, filter, bias, min_input, max_input, min_filter, max_filter,
          out_type=out_type, strides=strides, padding=padding,
          dilations=dilations, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if not isinstance(strides, (list, tuple)):
    raise TypeError(
        "Expected list for 'strides' argument to "
        "'quantized_depthwise_conv2d_with_bias' Op, not %r." % strides)
  strides = [_execute.make_int(_i, "strides") for _i in strides]
  padding = _execute.make_str(padding, "padding")
  if out_type is None:
    out_type = _dtypes.qint32
  out_type = _execute.make_type(out_type, "out_type")
  if dilations is None:
    dilations = [1, 1, 1, 1]
  if not isinstance(dilations, (list, tuple)):
    raise TypeError(
        "Expected list for 'dilations' argument to "
        "'quantized_depthwise_conv2d_with_bias' Op, not %r." % dilations)
  dilations = [_execute.make_int(_i, "dilations") for _i in dilations]
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "QuantizedDepthwiseConv2DWithBias", input=input, filter=filter,
                                            bias=bias, min_input=min_input,
                                            max_input=max_input,
                                            min_filter=min_filter,
                                            max_filter=max_filter,
                                            strides=strides, padding=padding,
                                            out_type=out_type,
                                            dilations=dilations, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("Tinput", _op._get_attr_type("Tinput"), "Tfilter",
              _op._get_attr_type("Tfilter"), "out_type",
              _op._get_attr_type("out_type"), "strides",
              _op.get_attr("strides"), "padding", _op.get_attr("padding"),
              "dilations", _op.get_attr("dilations"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "QuantizedDepthwiseConv2DWithBias", _inputs_flat, _attrs, _result)
  _result = _QuantizedDepthwiseConv2DWithBiasOutput._make(_result)
  return _result

QuantizedDepthwiseConv2DWithBias = tf_export("raw_ops.QuantizedDepthwiseConv2DWithBias")(_ops.to_raw_op(quantized_depthwise_conv2d_with_bias))


def quantized_depthwise_conv2d_with_bias_eager_fallback(input: Annotated[Any, TV_QuantizedDepthwiseConv2DWithBias_Tinput], filter: Annotated[Any, TV_QuantizedDepthwiseConv2DWithBias_Tfilter], bias: Annotated[Any, _atypes.Float32], min_input: Annotated[Any, _atypes.Float32], max_input: Annotated[Any, _atypes.Float32], min_filter: Annotated[Any, _atypes.Float32], max_filter: Annotated[Any, _atypes.Float32], strides, padding: str, out_type: TV_QuantizedDepthwiseConv2DWithBias_out_type, dilations, name, ctx):
  if not isinstance(strides, (list, tuple)):
    raise TypeError(
        "Expected list for 'strides' argument to "
        "'quantized_depthwise_conv2d_with_bias' Op, not %r." % strides)
  strides = [_execute.make_int(_i, "strides") for _i in strides]
  padding = _execute.make_str(padding, "padding")
  if out_type is None:
    out_type = _dtypes.qint32
  out_type = _execute.make_type(out_type, "out_type")
  if dilations is None:
    dilations = [1, 1, 1, 1]
  if not isinstance(dilations, (list, tuple)):
    raise TypeError(
        "Expected list for 'dilations' argument to "
        "'quantized_depthwise_conv2d_with_bias' Op, not %r." % dilations)
  dilations = [_execute.make_int(_i, "dilations") for _i in dilations]
  _attr_Tinput, (input,) = _execute.args_to_matching_eager([input], ctx, [_dtypes.qint8, _dtypes.quint8, _dtypes.qint32, _dtypes.qint16, _dtypes.quint16, ])
  _attr_Tfilter, (filter,) = _execute.args_to_matching_eager([filter], ctx, [_dtypes.qint8, _dtypes.quint8, _dtypes.qint32, _dtypes.qint16, _dtypes.quint16, ])
  bias = _ops.convert_to_tensor(bias, _dtypes.float32)
  min_input = _ops.convert_to_tensor(min_input, _dtypes.float32)
  max_input = _ops.convert_to_tensor(max_input, _dtypes.float32)
  min_filter = _ops.convert_to_tensor(min_filter, _dtypes.float32)
  max_filter = _ops.convert_to_tensor(max_filter, _dtypes.float32)
  _inputs_flat = [input, filter, bias, min_input, max_input, min_filter, max_filter]
  _attrs = ("Tinput", _attr_Tinput, "Tfilter", _attr_Tfilter, "out_type",
  out_type, "strides", strides, "padding", padding, "dilations", dilations)
  _result = _execute.execute(b"QuantizedDepthwiseConv2DWithBias", 3,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "QuantizedDepthwiseConv2DWithBias", _inputs_flat, _attrs, _result)
  _result = _QuantizedDepthwiseConv2DWithBiasOutput._make(_result)
  return _result

_QuantizedDepthwiseConv2DWithBiasAndReluOutput = collections.namedtuple(
    "QuantizedDepthwiseConv2DWithBiasAndRelu",
    ["output", "min_output", "max_output"])


TV_QuantizedDepthwiseConv2DWithBiasAndRelu_Tinput = TypeVar("TV_QuantizedDepthwiseConv2DWithBiasAndRelu_Tinput", _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8)
TV_QuantizedDepthwiseConv2DWithBiasAndRelu_Tfilter = TypeVar("TV_QuantizedDepthwiseConv2DWithBiasAndRelu_Tfilter", _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8)
TV_QuantizedDepthwiseConv2DWithBiasAndRelu_out_type = TypeVar("TV_QuantizedDepthwiseConv2DWithBiasAndRelu_out_type", _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8)

def quantized_depthwise_conv2d_with_bias_and_relu(input: Annotated[Any, TV_QuantizedDepthwiseConv2DWithBiasAndRelu_Tinput], filter: Annotated[Any, TV_QuantizedDepthwiseConv2DWithBiasAndRelu_Tfilter], bias: Annotated[Any, _atypes.Float32], min_input: Annotated[Any, _atypes.Float32], max_input: Annotated[Any, _atypes.Float32], min_filter: Annotated[Any, _atypes.Float32], max_filter: Annotated[Any, _atypes.Float32], strides, padding: str, out_type:TV_QuantizedDepthwiseConv2DWithBiasAndRelu_out_type=_dtypes.qint32, dilations=[1, 1, 1, 1], padding_list=[], name=None):
  r"""Computes quantized depthwise Conv2D with Bias and Relu.

  Args:
    input: A `Tensor`. Must be one of the following types: `qint8`, `quint8`, `qint32`, `qint16`, `quint16`.
      The original input tensor.
    filter: A `Tensor`. Must be one of the following types: `qint8`, `quint8`, `qint32`, `qint16`, `quint16`.
      The original filter tensor.
    bias: A `Tensor` of type `float32`. The original bias tensor.
    min_input: A `Tensor` of type `float32`.
      The float value that the minimum quantized input value represents.
    max_input: A `Tensor` of type `float32`.
      The float value that the maximum quantized input value represents.
    min_filter: A `Tensor` of type `float32`.
      The float value that the minimum quantized filter value represents.
    max_filter: A `Tensor` of type `float32`.
      The float value that the maximum quantized filter value represents.
    strides: A list of `ints`. List of stride values.
    padding: A `string` from: `"SAME", "VALID"`.
    out_type: An optional `tf.DType` from: `tf.qint8, tf.quint8, tf.qint32, tf.qint16, tf.quint16`. Defaults to `tf.qint32`.
      The type of the output.
    dilations: An optional list of `ints`. Defaults to `[1, 1, 1, 1]`.
      List of dilation values.
    padding_list: An optional list of `ints`. Defaults to `[]`.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (output, min_output, max_output).

    output: A `Tensor` of type `out_type`.
    min_output: A `Tensor` of type `float32`.
    max_output: A `Tensor` of type `float32`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "QuantizedDepthwiseConv2DWithBiasAndRelu", name, input, filter,
        bias, min_input, max_input, min_filter, max_filter, "out_type",
        out_type, "strides", strides, "padding", padding, "dilations",
        dilations, "padding_list", padding_list)
      _result = _QuantizedDepthwiseConv2DWithBiasAndReluOutput._make(_result)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return quantized_depthwise_conv2d_with_bias_and_relu_eager_fallback(
          input, filter, bias, min_input, max_input, min_filter, max_filter,
          out_type=out_type, strides=strides, padding=padding,
          dilations=dilations, padding_list=padding_list, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if not isinstance(strides, (list, tuple)):
    raise TypeError(
        "Expected list for 'strides' argument to "
        "'quantized_depthwise_conv2d_with_bias_and_relu' Op, not %r." % strides)
  strides = [_execute.make_int(_i, "strides") for _i in strides]
  padding = _execute.make_str(padding, "padding")
  if out_type is None:
    out_type = _dtypes.qint32
  out_type = _execute.make_type(out_type, "out_type")
  if dilations is None:
    dilations = [1, 1, 1, 1]
  if not isinstance(dilations, (list, tuple)):
    raise TypeError(
        "Expected list for 'dilations' argument to "
        "'quantized_depthwise_conv2d_with_bias_and_relu' Op, not %r." % dilations)
  dilations = [_execute.make_int(_i, "dilations") for _i in dilations]
  if padding_list is None:
    padding_list = []
  if not isinstance(padding_list, (list, tuple)):
    raise TypeError(
        "Expected list for 'padding_list' argument to "
        "'quantized_depthwise_conv2d_with_bias_and_relu' Op, not %r." % padding_list)
  padding_list = [_execute.make_int(_i, "padding_list") for _i in padding_list]
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "QuantizedDepthwiseConv2DWithBiasAndRelu", input=input, filter=filter,
                                                   bias=bias,
                                                   min_input=min_input,
                                                   max_input=max_input,
                                                   min_filter=min_filter,
                                                   max_filter=max_filter,
                                                   strides=strides,
                                                   padding=padding,
                                                   out_type=out_type,
                                                   dilations=dilations,
                                                   padding_list=padding_list,
                                                   name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("Tinput", _op._get_attr_type("Tinput"), "Tfilter",
              _op._get_attr_type("Tfilter"), "out_type",
              _op._get_attr_type("out_type"), "strides",
              _op.get_attr("strides"), "padding", _op.get_attr("padding"),
              "dilations", _op.get_attr("dilations"), "padding_list",
              _op.get_attr("padding_list"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "QuantizedDepthwiseConv2DWithBiasAndRelu", _inputs_flat, _attrs, _result)
  _result = _QuantizedDepthwiseConv2DWithBiasAndReluOutput._make(_result)
  return _result

QuantizedDepthwiseConv2DWithBiasAndRelu = tf_export("raw_ops.QuantizedDepthwiseConv2DWithBiasAndRelu")(_ops.to_raw_op(quantized_depthwise_conv2d_with_bias_and_relu))


def quantized_depthwise_conv2d_with_bias_and_relu_eager_fallback(input: Annotated[Any, TV_QuantizedDepthwiseConv2DWithBiasAndRelu_Tinput], filter: Annotated[Any, TV_QuantizedDepthwiseConv2DWithBiasAndRelu_Tfilter], bias: Annotated[Any, _atypes.Float32], min_input: Annotated[Any, _atypes.Float32], max_input: Annotated[Any, _atypes.Float32], min_filter: Annotated[Any, _atypes.Float32], max_filter: Annotated[Any, _atypes.Float32], strides, padding: str, out_type: TV_QuantizedDepthwiseConv2DWithBiasAndRelu_out_type, dilations, padding_list, name, ctx):
  if not isinstance(strides, (list, tuple)):
    raise TypeError(
        "Expected list for 'strides' argument to "
        "'quantized_depthwise_conv2d_with_bias_and_relu' Op, not %r." % strides)
  strides = [_execute.make_int(_i, "strides") for _i in strides]
  padding = _execute.make_str(padding, "padding")
  if out_type is None:
    out_type = _dtypes.qint32
  out_type = _execute.make_type(out_type, "out_type")
  if dilations is None:
    dilations = [1, 1, 1, 1]
  if not isinstance(dilations, (list, tuple)):
    raise TypeError(
        "Expected list for 'dilations' argument to "
        "'quantized_depthwise_conv2d_with_bias_and_relu' Op, not %r." % dilations)
  dilations = [_execute.make_int(_i, "dilations") for _i in dilations]
  if padding_list is None:
    padding_list = []
  if not isinstance(padding_list, (list, tuple)):
    raise TypeError(
        "Expected list for 'padding_list' argument to "
        "'quantized_depthwise_conv2d_with_bias_and_relu' Op, not %r." % padding_list)
  padding_list = [_execute.make_int(_i, "padding_list") for _i in padding_list]
  _attr_Tinput, (input,) = _execute.args_to_matching_eager([input], ctx, [_dtypes.qint8, _dtypes.quint8, _dtypes.qint32, _dtypes.qint16, _dtypes.quint16, ])
  _attr_Tfilter, (filter,) = _execute.args_to_matching_eager([filter], ctx, [_dtypes.qint8, _dtypes.quint8, _dtypes.qint32, _dtypes.qint16, _dtypes.quint16, ])
  bias = _ops.convert_to_tensor(bias, _dtypes.float32)
  min_input = _ops.convert_to_tensor(min_input, _dtypes.float32)
  max_input = _ops.convert_to_tensor(max_input, _dtypes.float32)
  min_filter = _ops.convert_to_tensor(min_filter, _dtypes.float32)
  max_filter = _ops.convert_to_tensor(max_filter, _dtypes.float32)
  _inputs_flat = [input, filter, bias, min_input, max_input, min_filter, max_filter]
  _attrs = ("Tinput", _attr_Tinput, "Tfilter", _attr_Tfilter, "out_type",
  out_type, "strides", strides, "padding", padding, "dilations", dilations,
  "padding_list", padding_list)
  _result = _execute.execute(b"QuantizedDepthwiseConv2DWithBiasAndRelu", 3,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "QuantizedDepthwiseConv2DWithBiasAndRelu", _inputs_flat, _attrs, _result)
  _result = _QuantizedDepthwiseConv2DWithBiasAndReluOutput._make(_result)
  return _result

_QuantizedDepthwiseConv2DWithBiasAndReluAndRequantizeOutput = collections.namedtuple(
    "QuantizedDepthwiseConv2DWithBiasAndReluAndRequantize",
    ["output", "min_output", "max_output"])


TV_QuantizedDepthwiseConv2DWithBiasAndReluAndRequantize_Tinput = TypeVar("TV_QuantizedDepthwiseConv2DWithBiasAndReluAndRequantize_Tinput", _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8)
TV_QuantizedDepthwiseConv2DWithBiasAndReluAndRequantize_Tfilter = TypeVar("TV_QuantizedDepthwiseConv2DWithBiasAndReluAndRequantize_Tfilter", _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8)
TV_QuantizedDepthwiseConv2DWithBiasAndReluAndRequantize_Tbias = TypeVar("TV_QuantizedDepthwiseConv2DWithBiasAndReluAndRequantize_Tbias", _atypes.Float32, _atypes.QInt32)
TV_QuantizedDepthwiseConv2DWithBiasAndReluAndRequantize_out_type = TypeVar("TV_QuantizedDepthwiseConv2DWithBiasAndReluAndRequantize_out_type", _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8)

def quantized_depthwise_conv2d_with_bias_and_relu_and_requantize(input: Annotated[Any, TV_QuantizedDepthwiseConv2DWithBiasAndReluAndRequantize_Tinput], filter: Annotated[Any, TV_QuantizedDepthwiseConv2DWithBiasAndReluAndRequantize_Tfilter], bias: Annotated[Any, TV_QuantizedDepthwiseConv2DWithBiasAndReluAndRequantize_Tbias], min_input: Annotated[Any, _atypes.Float32], max_input: Annotated[Any, _atypes.Float32], min_filter: Annotated[Any, _atypes.Float32], max_filter: Annotated[Any, _atypes.Float32], min_freezed_output: Annotated[Any, _atypes.Float32], max_freezed_output: Annotated[Any, _atypes.Float32], strides, padding: str, out_type:TV_QuantizedDepthwiseConv2DWithBiasAndReluAndRequantize_out_type=_dtypes.quint8, dilations=[1, 1, 1, 1], padding_list=[], name=None):
  r"""Computes quantized depthwise Conv2D with Bias, Relu and Requantize.

  Args:
    input: A `Tensor`. Must be one of the following types: `qint8`, `quint8`, `qint32`, `qint16`, `quint16`.
      The original input tensor.
    filter: A `Tensor`. Must be one of the following types: `qint8`, `quint8`, `qint32`, `qint16`, `quint16`.
      The original filter tensor.
    bias: A `Tensor`. Must be one of the following types: `float32`, `qint32`.
      The original bias tensor.
    min_input: A `Tensor` of type `float32`.
      The float value that the minimum quantized input value represents.
    max_input: A `Tensor` of type `float32`.
      The float value that the maximum quantized input value represents.
    min_filter: A `Tensor` of type `float32`.
      The float value that the minimum quantized filter value represents.
    max_filter: A `Tensor` of type `float32`.
      The float value that the maximum quantized filter value represents.
    min_freezed_output: A `Tensor` of type `float32`.
      The minimum float value of the output tensor.
    max_freezed_output: A `Tensor` of type `float32`.
      The maximum float value of the output tensor.
    strides: A list of `ints`. List of stride values.
    padding: A `string` from: `"SAME", "VALID"`.
    out_type: An optional `tf.DType` from: `tf.qint8, tf.quint8, tf.qint32, tf.qint16, tf.quint16`. Defaults to `tf.quint8`.
      The type of the output.
    dilations: An optional list of `ints`. Defaults to `[1, 1, 1, 1]`.
      List of dilation values.
    padding_list: An optional list of `ints`. Defaults to `[]`.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (output, min_output, max_output).

    output: A `Tensor` of type `out_type`.
    min_output: A `Tensor` of type `float32`.
    max_output: A `Tensor` of type `float32`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "QuantizedDepthwiseConv2DWithBiasAndReluAndRequantize", name,
        input, filter, bias, min_input, max_input, min_filter, max_filter,
        min_freezed_output, max_freezed_output, "out_type", out_type,
        "strides", strides, "padding", padding, "dilations", dilations,
        "padding_list", padding_list)
      _result = _QuantizedDepthwiseConv2DWithBiasAndReluAndRequantizeOutput._make(_result)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return quantized_depthwise_conv2d_with_bias_and_relu_and_requantize_eager_fallback(
          input, filter, bias, min_input, max_input, min_filter, max_filter,
          min_freezed_output, max_freezed_output, out_type=out_type,
          strides=strides, padding=padding, dilations=dilations,
          padding_list=padding_list, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if not isinstance(strides, (list, tuple)):
    raise TypeError(
        "Expected list for 'strides' argument to "
        "'quantized_depthwise_conv2d_with_bias_and_relu_and_requantize' Op, not %r." % strides)
  strides = [_execute.make_int(_i, "strides") for _i in strides]
  padding = _execute.make_str(padding, "padding")
  if out_type is None:
    out_type = _dtypes.quint8
  out_type = _execute.make_type(out_type, "out_type")
  if dilations is None:
    dilations = [1, 1, 1, 1]
  if not isinstance(dilations, (list, tuple)):
    raise TypeError(
        "Expected list for 'dilations' argument to "
        "'quantized_depthwise_conv2d_with_bias_and_relu_and_requantize' Op, not %r." % dilations)
  dilations = [_execute.make_int(_i, "dilations") for _i in dilations]
  if padding_list is None:
    padding_list = []
  if not isinstance(padding_list, (list, tuple)):
    raise TypeError(
        "Expected list for 'padding_list' argument to "
        "'quantized_depthwise_conv2d_with_bias_and_relu_and_requantize' Op, not %r." % padding_list)
  padding_list = [_execute.make_int(_i, "padding_list") for _i in padding_list]
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "QuantizedDepthwiseConv2DWithBiasAndReluAndRequantize", input=input,
                                                                filter=filter,
                                                                bias=bias,
                                                                min_input=min_input,
                                                                max_input=max_input,
                                                                min_filter=min_filter,
                                                                max_filter=max_filter,
                                                                min_freezed_output=min_freezed_output,
                                                                max_freezed_output=max_freezed_output,
                                                                strides=strides,
                                                                padding=padding,
                                                                out_type=out_type,
                                                                dilations=dilations,
                                                                padding_list=padding_list,
                                                                name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("Tinput", _op._get_attr_type("Tinput"), "Tfilter",
              _op._get_attr_type("Tfilter"), "Tbias",
              _op._get_attr_type("Tbias"), "out_type",
              _op._get_attr_type("out_type"), "strides",
              _op.get_attr("strides"), "padding", _op.get_attr("padding"),
              "dilations", _op.get_attr("dilations"), "padding_list",
              _op.get_attr("padding_list"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "QuantizedDepthwiseConv2DWithBiasAndReluAndRequantize", _inputs_flat, _attrs, _result)
  _result = _QuantizedDepthwiseConv2DWithBiasAndReluAndRequantizeOutput._make(_result)
  return _result

QuantizedDepthwiseConv2DWithBiasAndReluAndRequantize = tf_export("raw_ops.QuantizedDepthwiseConv2DWithBiasAndReluAndRequantize")(_ops.to_raw_op(quantized_depthwise_conv2d_with_bias_and_relu_and_requantize))


def quantized_depthwise_conv2d_with_bias_and_relu_and_requantize_eager_fallback(input: Annotated[Any, TV_QuantizedDepthwiseConv2DWithBiasAndReluAndRequantize_Tinput], filter: Annotated[Any, TV_QuantizedDepthwiseConv2DWithBiasAndReluAndRequantize_Tfilter], bias: Annotated[Any, TV_QuantizedDepthwiseConv2DWithBiasAndReluAndRequantize_Tbias], min_input: Annotated[Any, _atypes.Float32], max_input: Annotated[Any, _atypes.Float32], min_filter: Annotated[Any, _atypes.Float32], max_filter: Annotated[Any, _atypes.Float32], min_freezed_output: Annotated[Any, _atypes.Float32], max_freezed_output: Annotated[Any, _atypes.Float32], strides, padding: str, out_type: TV_QuantizedDepthwiseConv2DWithBiasAndReluAndRequantize_out_type, dilations, padding_list, name, ctx):
  if not isinstance(strides, (list, tuple)):
    raise TypeError(
        "Expected list for 'strides' argument to "
        "'quantized_depthwise_conv2d_with_bias_and_relu_and_requantize' Op, not %r." % strides)
  strides = [_execute.make_int(_i, "strides") for _i in strides]
  padding = _execute.make_str(padding, "padding")
  if out_type is None:
    out_type = _dtypes.quint8
  out_type = _execute.make_type(out_type, "out_type")
  if dilations is None:
    dilations = [1, 1, 1, 1]
  if not isinstance(dilations, (list, tuple)):
    raise TypeError(
        "Expected list for 'dilations' argument to "
        "'quantized_depthwise_conv2d_with_bias_and_relu_and_requantize' Op, not %r." % dilations)
  dilations = [_execute.make_int(_i, "dilations") for _i in dilations]
  if padding_list is None:
    padding_list = []
  if not isinstance(padding_list, (list, tuple)):
    raise TypeError(
        "Expected list for 'padding_list' argument to "
        "'quantized_depthwise_conv2d_with_bias_and_relu_and_requantize' Op, not %r." % padding_list)
  padding_list = [_execute.make_int(_i, "padding_list") for _i in padding_list]
  _attr_Tinput, (input,) = _execute.args_to_matching_eager([input], ctx, [_dtypes.qint8, _dtypes.quint8, _dtypes.qint32, _dtypes.qint16, _dtypes.quint16, ])
  _attr_Tfilter, (filter,) = _execute.args_to_matching_eager([filter], ctx, [_dtypes.qint8, _dtypes.quint8, _dtypes.qint32, _dtypes.qint16, _dtypes.quint16, ])
  _attr_Tbias, (bias,) = _execute.args_to_matching_eager([bias], ctx, [_dtypes.float32, _dtypes.qint32, ])
  min_input = _ops.convert_to_tensor(min_input, _dtypes.float32)
  max_input = _ops.convert_to_tensor(max_input, _dtypes.float32)
  min_filter = _ops.convert_to_tensor(min_filter, _dtypes.float32)
  max_filter = _ops.convert_to_tensor(max_filter, _dtypes.float32)
  min_freezed_output = _ops.convert_to_tensor(min_freezed_output, _dtypes.float32)
  max_freezed_output = _ops.convert_to_tensor(max_freezed_output, _dtypes.float32)
  _inputs_flat = [input, filter, bias, min_input, max_input, min_filter, max_filter, min_freezed_output, max_freezed_output]
  _attrs = ("Tinput", _attr_Tinput, "Tfilter", _attr_Tfilter, "Tbias",
  _attr_Tbias, "out_type", out_type, "strides", strides, "padding", padding,
  "dilations", dilations, "padding_list", padding_list)
  _result = _execute.execute(b"QuantizedDepthwiseConv2DWithBiasAndReluAndRequantize",
                             3, inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "QuantizedDepthwiseConv2DWithBiasAndReluAndRequantize", _inputs_flat, _attrs, _result)
  _result = _QuantizedDepthwiseConv2DWithBiasAndReluAndRequantizeOutput._make(_result)
  return _result

_QuantizedMatMulWithBiasOutput = collections.namedtuple(
    "QuantizedMatMulWithBias",
    ["out", "min_out", "max_out"])


TV_QuantizedMatMulWithBias_T1 = TypeVar("TV_QuantizedMatMulWithBias_T1", _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8)
TV_QuantizedMatMulWithBias_T2 = TypeVar("TV_QuantizedMatMulWithBias_T2", _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8)
TV_QuantizedMatMulWithBias_Tbias = TypeVar("TV_QuantizedMatMulWithBias_Tbias", _atypes.Float32, _atypes.QInt32)
TV_QuantizedMatMulWithBias_Toutput = TypeVar("TV_QuantizedMatMulWithBias_Toutput", _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8)

def quantized_mat_mul_with_bias(a: Annotated[Any, TV_QuantizedMatMulWithBias_T1], b: Annotated[Any, TV_QuantizedMatMulWithBias_T2], bias: Annotated[Any, TV_QuantizedMatMulWithBias_Tbias], min_a: Annotated[Any, _atypes.Float32], max_a: Annotated[Any, _atypes.Float32], min_b: Annotated[Any, _atypes.Float32], max_b: Annotated[Any, _atypes.Float32], Toutput:TV_QuantizedMatMulWithBias_Toutput=_dtypes.qint32, transpose_a:bool=False, transpose_b:bool=False, input_quant_mode:str="MIN_FIRST", name=None):
  r"""Performs a quantized matrix multiplication of `a` by the matrix `b` with bias
add.

  The inputs must be two-dimensional matrices and 1D bias vector. And the inner
  dimension of `a` (after being transposed if `transpose_a` is non-zero) must
  match the outer dimension of `b` (after being transposed if `transposed_b` is
  non-zero). Then do broadcast add operation with bias values on the matrix
  multiplication result. The bias size must match inner dimension of `b`.

  Args:
    a: A `Tensor`. Must be one of the following types: `qint8`, `quint8`, `qint32`, `qint16`, `quint16`.
      A matrix to be multiplied. Must be a two-dimensional tensor of type `quint8`.
    b: A `Tensor`. Must be one of the following types: `qint8`, `quint8`, `qint32`, `qint16`, `quint16`.
      A matrix to be multiplied and must be a two-dimensional tensor of type `qint8`.
    bias: A `Tensor`. Must be one of the following types: `float32`, `qint32`.
      A 1D bias tensor with size matching inner dimension of `b` (after being
      transposed if `transposed_b` is non-zero).
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
    input_quant_mode: An optional `string` from: `"MIN_FIRST", "SCALED"`. Defaults to `"MIN_FIRST"`.
      Input data quantization mode. Either MIN_FIRST(default) or SCALED.
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
        _ctx, "QuantizedMatMulWithBias", name, a, b, bias, min_a, max_a,
        min_b, max_b, "Toutput", Toutput, "transpose_a", transpose_a,
        "transpose_b", transpose_b, "input_quant_mode", input_quant_mode)
      _result = _QuantizedMatMulWithBiasOutput._make(_result)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return quantized_mat_mul_with_bias_eager_fallback(
          a, b, bias, min_a, max_a, min_b, max_b, Toutput=Toutput,
          transpose_a=transpose_a, transpose_b=transpose_b,
          input_quant_mode=input_quant_mode, name=name, ctx=_ctx)
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
  if input_quant_mode is None:
    input_quant_mode = "MIN_FIRST"
  input_quant_mode = _execute.make_str(input_quant_mode, "input_quant_mode")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "QuantizedMatMulWithBias", a=a, b=b, bias=bias, min_a=min_a,
                                   max_a=max_a, min_b=min_b, max_b=max_b,
                                   Toutput=Toutput, transpose_a=transpose_a,
                                   transpose_b=transpose_b,
                                   input_quant_mode=input_quant_mode,
                                   name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T1", _op._get_attr_type("T1"), "T2", _op._get_attr_type("T2"),
              "Tbias", _op._get_attr_type("Tbias"), "Toutput",
              _op._get_attr_type("Toutput"), "transpose_a",
              _op._get_attr_bool("transpose_a"), "transpose_b",
              _op._get_attr_bool("transpose_b"), "input_quant_mode",
              _op.get_attr("input_quant_mode"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "QuantizedMatMulWithBias", _inputs_flat, _attrs, _result)
  _result = _QuantizedMatMulWithBiasOutput._make(_result)
  return _result

QuantizedMatMulWithBias = tf_export("raw_ops.QuantizedMatMulWithBias")(_ops.to_raw_op(quantized_mat_mul_with_bias))


def quantized_mat_mul_with_bias_eager_fallback(a: Annotated[Any, TV_QuantizedMatMulWithBias_T1], b: Annotated[Any, TV_QuantizedMatMulWithBias_T2], bias: Annotated[Any, TV_QuantizedMatMulWithBias_Tbias], min_a: Annotated[Any, _atypes.Float32], max_a: Annotated[Any, _atypes.Float32], min_b: Annotated[Any, _atypes.Float32], max_b: Annotated[Any, _atypes.Float32], Toutput: TV_QuantizedMatMulWithBias_Toutput, transpose_a: bool, transpose_b: bool, input_quant_mode: str, name, ctx):
  if Toutput is None:
    Toutput = _dtypes.qint32
  Toutput = _execute.make_type(Toutput, "Toutput")
  if transpose_a is None:
    transpose_a = False
  transpose_a = _execute.make_bool(transpose_a, "transpose_a")
  if transpose_b is None:
    transpose_b = False
  transpose_b = _execute.make_bool(transpose_b, "transpose_b")
  if input_quant_mode is None:
    input_quant_mode = "MIN_FIRST"
  input_quant_mode = _execute.make_str(input_quant_mode, "input_quant_mode")
  _attr_T1, (a,) = _execute.args_to_matching_eager([a], ctx, [_dtypes.qint8, _dtypes.quint8, _dtypes.qint32, _dtypes.qint16, _dtypes.quint16, ])
  _attr_T2, (b,) = _execute.args_to_matching_eager([b], ctx, [_dtypes.qint8, _dtypes.quint8, _dtypes.qint32, _dtypes.qint16, _dtypes.quint16, ])
  _attr_Tbias, (bias,) = _execute.args_to_matching_eager([bias], ctx, [_dtypes.float32, _dtypes.qint32, ])
  min_a = _ops.convert_to_tensor(min_a, _dtypes.float32)
  max_a = _ops.convert_to_tensor(max_a, _dtypes.float32)
  min_b = _ops.convert_to_tensor(min_b, _dtypes.float32)
  max_b = _ops.convert_to_tensor(max_b, _dtypes.float32)
  _inputs_flat = [a, b, bias, min_a, max_a, min_b, max_b]
  _attrs = ("T1", _attr_T1, "T2", _attr_T2, "Tbias", _attr_Tbias, "Toutput",
  Toutput, "transpose_a", transpose_a, "transpose_b", transpose_b,
  "input_quant_mode", input_quant_mode)
  _result = _execute.execute(b"QuantizedMatMulWithBias", 3,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "QuantizedMatMulWithBias", _inputs_flat, _attrs, _result)
  _result = _QuantizedMatMulWithBiasOutput._make(_result)
  return _result


TV_QuantizedMatMulWithBiasAndDequantize_T1 = TypeVar("TV_QuantizedMatMulWithBiasAndDequantize_T1", _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8)
TV_QuantizedMatMulWithBiasAndDequantize_T2 = TypeVar("TV_QuantizedMatMulWithBiasAndDequantize_T2", _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8)
TV_QuantizedMatMulWithBiasAndDequantize_Tbias = TypeVar("TV_QuantizedMatMulWithBiasAndDequantize_Tbias", _atypes.Float32, _atypes.QInt32)
TV_QuantizedMatMulWithBiasAndDequantize_Toutput = TypeVar("TV_QuantizedMatMulWithBiasAndDequantize_Toutput", bound=_atypes.Float32)

def quantized_mat_mul_with_bias_and_dequantize(a: Annotated[Any, TV_QuantizedMatMulWithBiasAndDequantize_T1], b: Annotated[Any, TV_QuantizedMatMulWithBiasAndDequantize_T2], bias: Annotated[Any, TV_QuantizedMatMulWithBiasAndDequantize_Tbias], min_a: Annotated[Any, _atypes.Float32], max_a: Annotated[Any, _atypes.Float32], min_b: Annotated[Any, _atypes.Float32], max_b: Annotated[Any, _atypes.Float32], min_freezed_output: Annotated[Any, _atypes.Float32], max_freezed_output: Annotated[Any, _atypes.Float32], Toutput: TV_QuantizedMatMulWithBiasAndDequantize_Toutput, transpose_a:bool=False, transpose_b:bool=False, input_quant_mode:str="MIN_FIRST", name=None) -> Annotated[Any, TV_QuantizedMatMulWithBiasAndDequantize_Toutput]:
  r"""TODO: add doc.

  Args:
    a: A `Tensor`. Must be one of the following types: `qint8`, `quint8`, `qint32`, `qint16`, `quint16`.
    b: A `Tensor`. Must be one of the following types: `qint8`, `quint8`, `qint32`, `qint16`, `quint16`.
    bias: A `Tensor`. Must be one of the following types: `float32`, `qint32`.
    min_a: A `Tensor` of type `float32`.
    max_a: A `Tensor` of type `float32`.
    min_b: A `Tensor` of type `float32`.
    max_b: A `Tensor` of type `float32`.
    min_freezed_output: A `Tensor` of type `float32`.
    max_freezed_output: A `Tensor` of type `float32`.
    Toutput: A `tf.DType` from: `tf.float32`.
    transpose_a: An optional `bool`. Defaults to `False`.
    transpose_b: An optional `bool`. Defaults to `False`.
    input_quant_mode: An optional `string` from: `"MIN_FIRST", "SCALED"`. Defaults to `"MIN_FIRST"`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `Toutput`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "QuantizedMatMulWithBiasAndDequantize", name, a, b, bias, min_a,
        max_a, min_b, max_b, min_freezed_output, max_freezed_output,
        "Toutput", Toutput, "transpose_a", transpose_a, "transpose_b",
        transpose_b, "input_quant_mode", input_quant_mode)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return quantized_mat_mul_with_bias_and_dequantize_eager_fallback(
          a, b, bias, min_a, max_a, min_b, max_b, min_freezed_output,
          max_freezed_output, Toutput=Toutput, transpose_a=transpose_a,
          transpose_b=transpose_b, input_quant_mode=input_quant_mode,
          name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  Toutput = _execute.make_type(Toutput, "Toutput")
  if transpose_a is None:
    transpose_a = False
  transpose_a = _execute.make_bool(transpose_a, "transpose_a")
  if transpose_b is None:
    transpose_b = False
  transpose_b = _execute.make_bool(transpose_b, "transpose_b")
  if input_quant_mode is None:
    input_quant_mode = "MIN_FIRST"
  input_quant_mode = _execute.make_str(input_quant_mode, "input_quant_mode")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "QuantizedMatMulWithBiasAndDequantize", a=a, b=b, bias=bias,
                                                min_a=min_a, max_a=max_a,
                                                min_b=min_b, max_b=max_b,
                                                min_freezed_output=min_freezed_output,
                                                max_freezed_output=max_freezed_output,
                                                Toutput=Toutput,
                                                transpose_a=transpose_a,
                                                transpose_b=transpose_b,
                                                input_quant_mode=input_quant_mode,
                                                name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T1", _op._get_attr_type("T1"), "T2", _op._get_attr_type("T2"),
              "Tbias", _op._get_attr_type("Tbias"), "Toutput",
              _op._get_attr_type("Toutput"), "transpose_a",
              _op._get_attr_bool("transpose_a"), "transpose_b",
              _op._get_attr_bool("transpose_b"), "input_quant_mode",
              _op.get_attr("input_quant_mode"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "QuantizedMatMulWithBiasAndDequantize", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

QuantizedMatMulWithBiasAndDequantize = tf_export("raw_ops.QuantizedMatMulWithBiasAndDequantize")(_ops.to_raw_op(quantized_mat_mul_with_bias_and_dequantize))


def quantized_mat_mul_with_bias_and_dequantize_eager_fallback(a: Annotated[Any, TV_QuantizedMatMulWithBiasAndDequantize_T1], b: Annotated[Any, TV_QuantizedMatMulWithBiasAndDequantize_T2], bias: Annotated[Any, TV_QuantizedMatMulWithBiasAndDequantize_Tbias], min_a: Annotated[Any, _atypes.Float32], max_a: Annotated[Any, _atypes.Float32], min_b: Annotated[Any, _atypes.Float32], max_b: Annotated[Any, _atypes.Float32], min_freezed_output: Annotated[Any, _atypes.Float32], max_freezed_output: Annotated[Any, _atypes.Float32], Toutput: TV_QuantizedMatMulWithBiasAndDequantize_Toutput, transpose_a: bool, transpose_b: bool, input_quant_mode: str, name, ctx) -> Annotated[Any, TV_QuantizedMatMulWithBiasAndDequantize_Toutput]:
  Toutput = _execute.make_type(Toutput, "Toutput")
  if transpose_a is None:
    transpose_a = False
  transpose_a = _execute.make_bool(transpose_a, "transpose_a")
  if transpose_b is None:
    transpose_b = False
  transpose_b = _execute.make_bool(transpose_b, "transpose_b")
  if input_quant_mode is None:
    input_quant_mode = "MIN_FIRST"
  input_quant_mode = _execute.make_str(input_quant_mode, "input_quant_mode")
  _attr_T1, (a,) = _execute.args_to_matching_eager([a], ctx, [_dtypes.qint8, _dtypes.quint8, _dtypes.qint32, _dtypes.qint16, _dtypes.quint16, ])
  _attr_T2, (b,) = _execute.args_to_matching_eager([b], ctx, [_dtypes.qint8, _dtypes.quint8, _dtypes.qint32, _dtypes.qint16, _dtypes.quint16, ])
  _attr_Tbias, (bias,) = _execute.args_to_matching_eager([bias], ctx, [_dtypes.float32, _dtypes.qint32, ])
  min_a = _ops.convert_to_tensor(min_a, _dtypes.float32)
  max_a = _ops.convert_to_tensor(max_a, _dtypes.float32)
  min_b = _ops.convert_to_tensor(min_b, _dtypes.float32)
  max_b = _ops.convert_to_tensor(max_b, _dtypes.float32)
  min_freezed_output = _ops.convert_to_tensor(min_freezed_output, _dtypes.float32)
  max_freezed_output = _ops.convert_to_tensor(max_freezed_output, _dtypes.float32)
  _inputs_flat = [a, b, bias, min_a, max_a, min_b, max_b, min_freezed_output, max_freezed_output]
  _attrs = ("T1", _attr_T1, "T2", _attr_T2, "Tbias", _attr_Tbias, "Toutput",
  Toutput, "transpose_a", transpose_a, "transpose_b", transpose_b,
  "input_quant_mode", input_quant_mode)
  _result = _execute.execute(b"QuantizedMatMulWithBiasAndDequantize", 1,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "QuantizedMatMulWithBiasAndDequantize", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

_QuantizedMatMulWithBiasAndReluOutput = collections.namedtuple(
    "QuantizedMatMulWithBiasAndRelu",
    ["out", "min_out", "max_out"])


TV_QuantizedMatMulWithBiasAndRelu_T1 = TypeVar("TV_QuantizedMatMulWithBiasAndRelu_T1", _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8)
TV_QuantizedMatMulWithBiasAndRelu_T2 = TypeVar("TV_QuantizedMatMulWithBiasAndRelu_T2", _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8)
TV_QuantizedMatMulWithBiasAndRelu_Toutput = TypeVar("TV_QuantizedMatMulWithBiasAndRelu_Toutput", _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8)

def quantized_mat_mul_with_bias_and_relu(a: Annotated[Any, TV_QuantizedMatMulWithBiasAndRelu_T1], b: Annotated[Any, TV_QuantizedMatMulWithBiasAndRelu_T2], bias: Annotated[Any, _atypes.Float32], min_a: Annotated[Any, _atypes.Float32], max_a: Annotated[Any, _atypes.Float32], min_b: Annotated[Any, _atypes.Float32], max_b: Annotated[Any, _atypes.Float32], Toutput:TV_QuantizedMatMulWithBiasAndRelu_Toutput=_dtypes.qint32, transpose_a:bool=False, transpose_b:bool=False, input_quant_mode:str="MIN_FIRST", name=None):
  r"""Perform a quantized matrix multiplication of  `a` by the matrix `b` with bias
add and relu fusion.

  The inputs must be two-dimensional matrices and 1D bias vector. And the inner
  dimension of `a` (after being transposed if `transpose_a` is non-zero) must
  match the outer dimension of `b` (after being transposed if `transposed_b` is
  non-zero). Then do broadcast add operation with bias values on the matrix
  multiplication result. The bias size must match inner dimension of `b`. Then do
  relu activation to get non-negative result.

  Args:
    a: A `Tensor`. Must be one of the following types: `qint8`, `quint8`, `qint32`, `qint16`, `quint16`.
      A matrix to be multiplied. Must be a two-dimensional tensor of type `quint8`.
    b: A `Tensor`. Must be one of the following types: `qint8`, `quint8`, `qint32`, `qint16`, `quint16`.
      A matrix to be multiplied and must be a two-dimensional tensor of type `qint8`.
    bias: A `Tensor` of type `float32`.
      A 1D bias tensor with size matching with inner dimension of `b` (after being
      transposed if `transposed_b` is non-zero).
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
    input_quant_mode: An optional `string` from: `"MIN_FIRST", "SCALED"`. Defaults to `"MIN_FIRST"`.
      Input data quantization mode. Either MIN_FIRST(default) or SCALED.
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
        _ctx, "QuantizedMatMulWithBiasAndRelu", name, a, b, bias, min_a,
        max_a, min_b, max_b, "Toutput", Toutput, "transpose_a", transpose_a,
        "transpose_b", transpose_b, "input_quant_mode", input_quant_mode)
      _result = _QuantizedMatMulWithBiasAndReluOutput._make(_result)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return quantized_mat_mul_with_bias_and_relu_eager_fallback(
          a, b, bias, min_a, max_a, min_b, max_b, Toutput=Toutput,
          transpose_a=transpose_a, transpose_b=transpose_b,
          input_quant_mode=input_quant_mode, name=name, ctx=_ctx)
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
  if input_quant_mode is None:
    input_quant_mode = "MIN_FIRST"
  input_quant_mode = _execute.make_str(input_quant_mode, "input_quant_mode")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "QuantizedMatMulWithBiasAndRelu", a=a, b=b, bias=bias, min_a=min_a,
                                          max_a=max_a, min_b=min_b,
                                          max_b=max_b, Toutput=Toutput,
                                          transpose_a=transpose_a,
                                          transpose_b=transpose_b,
                                          input_quant_mode=input_quant_mode,
                                          name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T1", _op._get_attr_type("T1"), "T2", _op._get_attr_type("T2"),
              "Toutput", _op._get_attr_type("Toutput"), "transpose_a",
              _op._get_attr_bool("transpose_a"), "transpose_b",
              _op._get_attr_bool("transpose_b"), "input_quant_mode",
              _op.get_attr("input_quant_mode"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "QuantizedMatMulWithBiasAndRelu", _inputs_flat, _attrs, _result)
  _result = _QuantizedMatMulWithBiasAndReluOutput._make(_result)
  return _result

QuantizedMatMulWithBiasAndRelu = tf_export("raw_ops.QuantizedMatMulWithBiasAndRelu")(_ops.to_raw_op(quantized_mat_mul_with_bias_and_relu))


def quantized_mat_mul_with_bias_and_relu_eager_fallback(a: Annotated[Any, TV_QuantizedMatMulWithBiasAndRelu_T1], b: Annotated[Any, TV_QuantizedMatMulWithBiasAndRelu_T2], bias: Annotated[Any, _atypes.Float32], min_a: Annotated[Any, _atypes.Float32], max_a: Annotated[Any, _atypes.Float32], min_b: Annotated[Any, _atypes.Float32], max_b: Annotated[Any, _atypes.Float32], Toutput: TV_QuantizedMatMulWithBiasAndRelu_Toutput, transpose_a: bool, transpose_b: bool, input_quant_mode: str, name, ctx):
  if Toutput is None:
    Toutput = _dtypes.qint32
  Toutput = _execute.make_type(Toutput, "Toutput")
  if transpose_a is None:
    transpose_a = False
  transpose_a = _execute.make_bool(transpose_a, "transpose_a")
  if transpose_b is None:
    transpose_b = False
  transpose_b = _execute.make_bool(transpose_b, "transpose_b")
  if input_quant_mode is None:
    input_quant_mode = "MIN_FIRST"
  input_quant_mode = _execute.make_str(input_quant_mode, "input_quant_mode")
  _attr_T1, (a,) = _execute.args_to_matching_eager([a], ctx, [_dtypes.qint8, _dtypes.quint8, _dtypes.qint32, _dtypes.qint16, _dtypes.quint16, ])
  _attr_T2, (b,) = _execute.args_to_matching_eager([b], ctx, [_dtypes.qint8, _dtypes.quint8, _dtypes.qint32, _dtypes.qint16, _dtypes.quint16, ])
  bias = _ops.convert_to_tensor(bias, _dtypes.float32)
  min_a = _ops.convert_to_tensor(min_a, _dtypes.float32)
  max_a = _ops.convert_to_tensor(max_a, _dtypes.float32)
  min_b = _ops.convert_to_tensor(min_b, _dtypes.float32)
  max_b = _ops.convert_to_tensor(max_b, _dtypes.float32)
  _inputs_flat = [a, b, bias, min_a, max_a, min_b, max_b]
  _attrs = ("T1", _attr_T1, "T2", _attr_T2, "Toutput", Toutput, "transpose_a",
  transpose_a, "transpose_b", transpose_b, "input_quant_mode",
  input_quant_mode)
  _result = _execute.execute(b"QuantizedMatMulWithBiasAndRelu", 3,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "QuantizedMatMulWithBiasAndRelu", _inputs_flat, _attrs, _result)
  _result = _QuantizedMatMulWithBiasAndReluOutput._make(_result)
  return _result

_QuantizedMatMulWithBiasAndReluAndRequantizeOutput = collections.namedtuple(
    "QuantizedMatMulWithBiasAndReluAndRequantize",
    ["out", "min_out", "max_out"])


TV_QuantizedMatMulWithBiasAndReluAndRequantize_T1 = TypeVar("TV_QuantizedMatMulWithBiasAndReluAndRequantize_T1", _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8)
TV_QuantizedMatMulWithBiasAndReluAndRequantize_T2 = TypeVar("TV_QuantizedMatMulWithBiasAndReluAndRequantize_T2", _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8)
TV_QuantizedMatMulWithBiasAndReluAndRequantize_Tbias = TypeVar("TV_QuantizedMatMulWithBiasAndReluAndRequantize_Tbias", _atypes.Float32, _atypes.QInt32)
TV_QuantizedMatMulWithBiasAndReluAndRequantize_Toutput = TypeVar("TV_QuantizedMatMulWithBiasAndReluAndRequantize_Toutput", _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8)

def quantized_mat_mul_with_bias_and_relu_and_requantize(a: Annotated[Any, TV_QuantizedMatMulWithBiasAndReluAndRequantize_T1], b: Annotated[Any, TV_QuantizedMatMulWithBiasAndReluAndRequantize_T2], bias: Annotated[Any, TV_QuantizedMatMulWithBiasAndReluAndRequantize_Tbias], min_a: Annotated[Any, _atypes.Float32], max_a: Annotated[Any, _atypes.Float32], min_b: Annotated[Any, _atypes.Float32], max_b: Annotated[Any, _atypes.Float32], min_freezed_output: Annotated[Any, _atypes.Float32], max_freezed_output: Annotated[Any, _atypes.Float32], Toutput:TV_QuantizedMatMulWithBiasAndReluAndRequantize_Toutput=_dtypes.quint8, transpose_a:bool=False, transpose_b:bool=False, input_quant_mode:str="MIN_FIRST", name=None):
  r"""Perform a quantized matrix multiplication of  `a` by the matrix `b` with bias
add and relu and requantize fusion.

  The inputs must be two-dimensional matrices and 1D bias vector. And the inner
  dimension of `a` (after being transposed if `transpose_a` is non-zero) must
  match the outer dimension of `b` (after being transposed if `transposed_b` is
  non-zero). Then do broadcast add operation with bias values on the matrix
  multiplication result. The bias size must match inner dimension of `b`.  Then do
  relu activation to get non-negative result. Then do requantize operation to get
  final uint8 result.

  Args:
    a: A `Tensor`. Must be one of the following types: `qint8`, `quint8`, `qint32`, `qint16`, `quint16`.
      A matrix to be multiplied. Must be a two-dimensional tensor of type `quint8`.
    b: A `Tensor`. Must be one of the following types: `qint8`, `quint8`, `qint32`, `qint16`, `quint16`.
      A matrix to be multiplied and must be a two-dimensional tensor of type `qint8`.
    bias: A `Tensor`. Must be one of the following types: `float32`, `qint32`.
      A 1D bias tensor with size matching with inner dimension of `b` (after being
      transposed if `transposed_b` is non-zero).
    min_a: A `Tensor` of type `float32`.
      The float value that the lowest quantized `a` value represents.
    max_a: A `Tensor` of type `float32`.
      The float value that the highest quantized `a` value represents.
    min_b: A `Tensor` of type `float32`.
      The float value that the lowest quantized `b` value represents.
    max_b: A `Tensor` of type `float32`.
      The float value that the highest quantized `b` value represents.
    min_freezed_output: A `Tensor` of type `float32`.
      The float value that the highest quantized output value after requantize.
    max_freezed_output: A `Tensor` of type `float32`.
    Toutput: An optional `tf.DType` from: `tf.qint8, tf.quint8, tf.qint32, tf.qint16, tf.quint16`. Defaults to `tf.quint8`.
    transpose_a: An optional `bool`. Defaults to `False`.
      If true, `a` is transposed before multiplication.
    transpose_b: An optional `bool`. Defaults to `False`.
      If true, `b` is transposed before multiplication.
    input_quant_mode: An optional `string` from: `"MIN_FIRST", "SCALED"`. Defaults to `"MIN_FIRST"`.
      Input data quantization mode. Either MIN_FIRST(default) or SCALED.
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
        _ctx, "QuantizedMatMulWithBiasAndReluAndRequantize", name, a, b, bias,
        min_a, max_a, min_b, max_b, min_freezed_output, max_freezed_output,
        "Toutput", Toutput, "transpose_a", transpose_a, "transpose_b",
        transpose_b, "input_quant_mode", input_quant_mode)
      _result = _QuantizedMatMulWithBiasAndReluAndRequantizeOutput._make(_result)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return quantized_mat_mul_with_bias_and_relu_and_requantize_eager_fallback(
          a, b, bias, min_a, max_a, min_b, max_b, min_freezed_output,
          max_freezed_output, Toutput=Toutput, transpose_a=transpose_a,
          transpose_b=transpose_b, input_quant_mode=input_quant_mode,
          name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if Toutput is None:
    Toutput = _dtypes.quint8
  Toutput = _execute.make_type(Toutput, "Toutput")
  if transpose_a is None:
    transpose_a = False
  transpose_a = _execute.make_bool(transpose_a, "transpose_a")
  if transpose_b is None:
    transpose_b = False
  transpose_b = _execute.make_bool(transpose_b, "transpose_b")
  if input_quant_mode is None:
    input_quant_mode = "MIN_FIRST"
  input_quant_mode = _execute.make_str(input_quant_mode, "input_quant_mode")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "QuantizedMatMulWithBiasAndReluAndRequantize", a=a, b=b, bias=bias,
                                                       min_a=min_a,
                                                       max_a=max_a,
                                                       min_b=min_b,
                                                       max_b=max_b,
                                                       min_freezed_output=min_freezed_output,
                                                       max_freezed_output=max_freezed_output,
                                                       Toutput=Toutput,
                                                       transpose_a=transpose_a,
                                                       transpose_b=transpose_b,
                                                       input_quant_mode=input_quant_mode,
                                                       name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T1", _op._get_attr_type("T1"), "T2", _op._get_attr_type("T2"),
              "Tbias", _op._get_attr_type("Tbias"), "Toutput",
              _op._get_attr_type("Toutput"), "transpose_a",
              _op._get_attr_bool("transpose_a"), "transpose_b",
              _op._get_attr_bool("transpose_b"), "input_quant_mode",
              _op.get_attr("input_quant_mode"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "QuantizedMatMulWithBiasAndReluAndRequantize", _inputs_flat, _attrs, _result)
  _result = _QuantizedMatMulWithBiasAndReluAndRequantizeOutput._make(_result)
  return _result

QuantizedMatMulWithBiasAndReluAndRequantize = tf_export("raw_ops.QuantizedMatMulWithBiasAndReluAndRequantize")(_ops.to_raw_op(quantized_mat_mul_with_bias_and_relu_and_requantize))


def quantized_mat_mul_with_bias_and_relu_and_requantize_eager_fallback(a: Annotated[Any, TV_QuantizedMatMulWithBiasAndReluAndRequantize_T1], b: Annotated[Any, TV_QuantizedMatMulWithBiasAndReluAndRequantize_T2], bias: Annotated[Any, TV_QuantizedMatMulWithBiasAndReluAndRequantize_Tbias], min_a: Annotated[Any, _atypes.Float32], max_a: Annotated[Any, _atypes.Float32], min_b: Annotated[Any, _atypes.Float32], max_b: Annotated[Any, _atypes.Float32], min_freezed_output: Annotated[Any, _atypes.Float32], max_freezed_output: Annotated[Any, _atypes.Float32], Toutput: TV_QuantizedMatMulWithBiasAndReluAndRequantize_Toutput, transpose_a: bool, transpose_b: bool, input_quant_mode: str, name, ctx):
  if Toutput is None:
    Toutput = _dtypes.quint8
  Toutput = _execute.make_type(Toutput, "Toutput")
  if transpose_a is None:
    transpose_a = False
  transpose_a = _execute.make_bool(transpose_a, "transpose_a")
  if transpose_b is None:
    transpose_b = False
  transpose_b = _execute.make_bool(transpose_b, "transpose_b")
  if input_quant_mode is None:
    input_quant_mode = "MIN_FIRST"
  input_quant_mode = _execute.make_str(input_quant_mode, "input_quant_mode")
  _attr_T1, (a,) = _execute.args_to_matching_eager([a], ctx, [_dtypes.qint8, _dtypes.quint8, _dtypes.qint32, _dtypes.qint16, _dtypes.quint16, ])
  _attr_T2, (b,) = _execute.args_to_matching_eager([b], ctx, [_dtypes.qint8, _dtypes.quint8, _dtypes.qint32, _dtypes.qint16, _dtypes.quint16, ])
  _attr_Tbias, (bias,) = _execute.args_to_matching_eager([bias], ctx, [_dtypes.float32, _dtypes.qint32, ])
  min_a = _ops.convert_to_tensor(min_a, _dtypes.float32)
  max_a = _ops.convert_to_tensor(max_a, _dtypes.float32)
  min_b = _ops.convert_to_tensor(min_b, _dtypes.float32)
  max_b = _ops.convert_to_tensor(max_b, _dtypes.float32)
  min_freezed_output = _ops.convert_to_tensor(min_freezed_output, _dtypes.float32)
  max_freezed_output = _ops.convert_to_tensor(max_freezed_output, _dtypes.float32)
  _inputs_flat = [a, b, bias, min_a, max_a, min_b, max_b, min_freezed_output, max_freezed_output]
  _attrs = ("T1", _attr_T1, "T2", _attr_T2, "Tbias", _attr_Tbias, "Toutput",
  Toutput, "transpose_a", transpose_a, "transpose_b", transpose_b,
  "input_quant_mode", input_quant_mode)
  _result = _execute.execute(b"QuantizedMatMulWithBiasAndReluAndRequantize",
                             3, inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "QuantizedMatMulWithBiasAndReluAndRequantize", _inputs_flat, _attrs, _result)
  _result = _QuantizedMatMulWithBiasAndReluAndRequantizeOutput._make(_result)
  return _result

_QuantizedMatMulWithBiasAndRequantizeOutput = collections.namedtuple(
    "QuantizedMatMulWithBiasAndRequantize",
    ["out", "min_out", "max_out"])


TV_QuantizedMatMulWithBiasAndRequantize_T1 = TypeVar("TV_QuantizedMatMulWithBiasAndRequantize_T1", _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8)
TV_QuantizedMatMulWithBiasAndRequantize_T2 = TypeVar("TV_QuantizedMatMulWithBiasAndRequantize_T2", _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8)
TV_QuantizedMatMulWithBiasAndRequantize_Tbias = TypeVar("TV_QuantizedMatMulWithBiasAndRequantize_Tbias", _atypes.Float32, _atypes.QInt32)
TV_QuantizedMatMulWithBiasAndRequantize_Toutput = TypeVar("TV_QuantizedMatMulWithBiasAndRequantize_Toutput", _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8)

def quantized_mat_mul_with_bias_and_requantize(a: Annotated[Any, TV_QuantizedMatMulWithBiasAndRequantize_T1], b: Annotated[Any, TV_QuantizedMatMulWithBiasAndRequantize_T2], bias: Annotated[Any, TV_QuantizedMatMulWithBiasAndRequantize_Tbias], min_a: Annotated[Any, _atypes.Float32], max_a: Annotated[Any, _atypes.Float32], min_b: Annotated[Any, _atypes.Float32], max_b: Annotated[Any, _atypes.Float32], min_freezed_output: Annotated[Any, _atypes.Float32], max_freezed_output: Annotated[Any, _atypes.Float32], Toutput:TV_QuantizedMatMulWithBiasAndRequantize_Toutput=_dtypes.quint8, transpose_a:bool=False, transpose_b:bool=False, input_quant_mode:str="MIN_FIRST", name=None):
  r"""TODO: add doc.

  Args:
    a: A `Tensor`. Must be one of the following types: `qint8`, `quint8`, `qint32`, `qint16`, `quint16`.
    b: A `Tensor`. Must be one of the following types: `qint8`, `quint8`, `qint32`, `qint16`, `quint16`.
    bias: A `Tensor`. Must be one of the following types: `float32`, `qint32`.
    min_a: A `Tensor` of type `float32`.
    max_a: A `Tensor` of type `float32`.
    min_b: A `Tensor` of type `float32`.
    max_b: A `Tensor` of type `float32`.
    min_freezed_output: A `Tensor` of type `float32`.
    max_freezed_output: A `Tensor` of type `float32`.
    Toutput: An optional `tf.DType` from: `tf.qint8, tf.quint8, tf.qint32, tf.qint16, tf.quint16`. Defaults to `tf.quint8`.
    transpose_a: An optional `bool`. Defaults to `False`.
    transpose_b: An optional `bool`. Defaults to `False`.
    input_quant_mode: An optional `string` from: `"MIN_FIRST", "SCALED"`. Defaults to `"MIN_FIRST"`.
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
        _ctx, "QuantizedMatMulWithBiasAndRequantize", name, a, b, bias, min_a,
        max_a, min_b, max_b, min_freezed_output, max_freezed_output,
        "Toutput", Toutput, "transpose_a", transpose_a, "transpose_b",
        transpose_b, "input_quant_mode", input_quant_mode)
      _result = _QuantizedMatMulWithBiasAndRequantizeOutput._make(_result)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return quantized_mat_mul_with_bias_and_requantize_eager_fallback(
          a, b, bias, min_a, max_a, min_b, max_b, min_freezed_output,
          max_freezed_output, Toutput=Toutput, transpose_a=transpose_a,
          transpose_b=transpose_b, input_quant_mode=input_quant_mode,
          name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if Toutput is None:
    Toutput = _dtypes.quint8
  Toutput = _execute.make_type(Toutput, "Toutput")
  if transpose_a is None:
    transpose_a = False
  transpose_a = _execute.make_bool(transpose_a, "transpose_a")
  if transpose_b is None:
    transpose_b = False
  transpose_b = _execute.make_bool(transpose_b, "transpose_b")
  if input_quant_mode is None:
    input_quant_mode = "MIN_FIRST"
  input_quant_mode = _execute.make_str(input_quant_mode, "input_quant_mode")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "QuantizedMatMulWithBiasAndRequantize", a=a, b=b, bias=bias,
                                                min_a=min_a, max_a=max_a,
                                                min_b=min_b, max_b=max_b,
                                                min_freezed_output=min_freezed_output,
                                                max_freezed_output=max_freezed_output,
                                                Toutput=Toutput,
                                                transpose_a=transpose_a,
                                                transpose_b=transpose_b,
                                                input_quant_mode=input_quant_mode,
                                                name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T1", _op._get_attr_type("T1"), "T2", _op._get_attr_type("T2"),
              "Tbias", _op._get_attr_type("Tbias"), "Toutput",
              _op._get_attr_type("Toutput"), "transpose_a",
              _op._get_attr_bool("transpose_a"), "transpose_b",
              _op._get_attr_bool("transpose_b"), "input_quant_mode",
              _op.get_attr("input_quant_mode"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "QuantizedMatMulWithBiasAndRequantize", _inputs_flat, _attrs, _result)
  _result = _QuantizedMatMulWithBiasAndRequantizeOutput._make(_result)
  return _result

QuantizedMatMulWithBiasAndRequantize = tf_export("raw_ops.QuantizedMatMulWithBiasAndRequantize")(_ops.to_raw_op(quantized_mat_mul_with_bias_and_requantize))


def quantized_mat_mul_with_bias_and_requantize_eager_fallback(a: Annotated[Any, TV_QuantizedMatMulWithBiasAndRequantize_T1], b: Annotated[Any, TV_QuantizedMatMulWithBiasAndRequantize_T2], bias: Annotated[Any, TV_QuantizedMatMulWithBiasAndRequantize_Tbias], min_a: Annotated[Any, _atypes.Float32], max_a: Annotated[Any, _atypes.Float32], min_b: Annotated[Any, _atypes.Float32], max_b: Annotated[Any, _atypes.Float32], min_freezed_output: Annotated[Any, _atypes.Float32], max_freezed_output: Annotated[Any, _atypes.Float32], Toutput: TV_QuantizedMatMulWithBiasAndRequantize_Toutput, transpose_a: bool, transpose_b: bool, input_quant_mode: str, name, ctx):
  if Toutput is None:
    Toutput = _dtypes.quint8
  Toutput = _execute.make_type(Toutput, "Toutput")
  if transpose_a is None:
    transpose_a = False
  transpose_a = _execute.make_bool(transpose_a, "transpose_a")
  if transpose_b is None:
    transpose_b = False
  transpose_b = _execute.make_bool(transpose_b, "transpose_b")
  if input_quant_mode is None:
    input_quant_mode = "MIN_FIRST"
  input_quant_mode = _execute.make_str(input_quant_mode, "input_quant_mode")
  _attr_T1, (a,) = _execute.args_to_matching_eager([a], ctx, [_dtypes.qint8, _dtypes.quint8, _dtypes.qint32, _dtypes.qint16, _dtypes.quint16, ])
  _attr_T2, (b,) = _execute.args_to_matching_eager([b], ctx, [_dtypes.qint8, _dtypes.quint8, _dtypes.qint32, _dtypes.qint16, _dtypes.quint16, ])
  _attr_Tbias, (bias,) = _execute.args_to_matching_eager([bias], ctx, [_dtypes.float32, _dtypes.qint32, ])
  min_a = _ops.convert_to_tensor(min_a, _dtypes.float32)
  max_a = _ops.convert_to_tensor(max_a, _dtypes.float32)
  min_b = _ops.convert_to_tensor(min_b, _dtypes.float32)
  max_b = _ops.convert_to_tensor(max_b, _dtypes.float32)
  min_freezed_output = _ops.convert_to_tensor(min_freezed_output, _dtypes.float32)
  max_freezed_output = _ops.convert_to_tensor(max_freezed_output, _dtypes.float32)
  _inputs_flat = [a, b, bias, min_a, max_a, min_b, max_b, min_freezed_output, max_freezed_output]
  _attrs = ("T1", _attr_T1, "T2", _attr_T2, "Tbias", _attr_Tbias, "Toutput",
  Toutput, "transpose_a", transpose_a, "transpose_b", transpose_b,
  "input_quant_mode", input_quant_mode)
  _result = _execute.execute(b"QuantizedMatMulWithBiasAndRequantize", 3,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "QuantizedMatMulWithBiasAndRequantize", _inputs_flat, _attrs, _result)
  _result = _QuantizedMatMulWithBiasAndRequantizeOutput._make(_result)
  return _result

_QuantizedMaxPoolOutput = collections.namedtuple(
    "QuantizedMaxPool",
    ["output", "min_output", "max_output"])


TV_QuantizedMaxPool_T = TypeVar("TV_QuantizedMaxPool_T", _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8)

def quantized_max_pool(input: Annotated[Any, TV_QuantizedMaxPool_T], min_input: Annotated[Any, _atypes.Float32], max_input: Annotated[Any, _atypes.Float32], ksize, strides, padding: str, name=None):
  r"""Produces the max pool of the input tensor for quantized types.

  Args:
    input: A `Tensor`. Must be one of the following types: `qint8`, `quint8`, `qint32`, `qint16`, `quint16`.
      The 4D (batch x rows x cols x depth) Tensor to MaxReduce over.
    min_input: A `Tensor` of type `float32`.
      The float value that the lowest quantized input value represents.
    max_input: A `Tensor` of type `float32`.
      The float value that the highest quantized input value represents.
    ksize: A list of `ints`.
      The size of the window for each dimension of the input tensor.
      The length must be 4 to match the number of dimensions of the input.
    strides: A list of `ints`.
      The stride of the sliding window for each dimension of the input
      tensor. The length must be 4 to match the number of dimensions of the input.
    padding: A `string` from: `"SAME", "VALID"`.
      The type of padding algorithm to use.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (output, min_output, max_output).

    output: A `Tensor`. Has the same type as `input`.
    min_output: A `Tensor` of type `float32`.
    max_output: A `Tensor` of type `float32`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "QuantizedMaxPool", name, input, min_input, max_input, "ksize",
        ksize, "strides", strides, "padding", padding)
      _result = _QuantizedMaxPoolOutput._make(_result)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return quantized_max_pool_eager_fallback(
          input, min_input, max_input, ksize=ksize, strides=strides,
          padding=padding, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if not isinstance(ksize, (list, tuple)):
    raise TypeError(
        "Expected list for 'ksize' argument to "
        "'quantized_max_pool' Op, not %r." % ksize)
  ksize = [_execute.make_int(_i, "ksize") for _i in ksize]
  if not isinstance(strides, (list, tuple)):
    raise TypeError(
        "Expected list for 'strides' argument to "
        "'quantized_max_pool' Op, not %r." % strides)
  strides = [_execute.make_int(_i, "strides") for _i in strides]
  padding = _execute.make_str(padding, "padding")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "QuantizedMaxPool", input=input, min_input=min_input,
                            max_input=max_input, ksize=ksize, strides=strides,
                            padding=padding, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"), "ksize", _op.get_attr("ksize"),
              "strides", _op.get_attr("strides"), "padding",
              _op.get_attr("padding"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "QuantizedMaxPool", _inputs_flat, _attrs, _result)
  _result = _QuantizedMaxPoolOutput._make(_result)
  return _result

QuantizedMaxPool = tf_export("raw_ops.QuantizedMaxPool")(_ops.to_raw_op(quantized_max_pool))


def quantized_max_pool_eager_fallback(input: Annotated[Any, TV_QuantizedMaxPool_T], min_input: Annotated[Any, _atypes.Float32], max_input: Annotated[Any, _atypes.Float32], ksize, strides, padding: str, name, ctx):
  if not isinstance(ksize, (list, tuple)):
    raise TypeError(
        "Expected list for 'ksize' argument to "
        "'quantized_max_pool' Op, not %r." % ksize)
  ksize = [_execute.make_int(_i, "ksize") for _i in ksize]
  if not isinstance(strides, (list, tuple)):
    raise TypeError(
        "Expected list for 'strides' argument to "
        "'quantized_max_pool' Op, not %r." % strides)
  strides = [_execute.make_int(_i, "strides") for _i in strides]
  padding = _execute.make_str(padding, "padding")
  _attr_T, (input,) = _execute.args_to_matching_eager([input], ctx, [_dtypes.qint8, _dtypes.quint8, _dtypes.qint32, _dtypes.qint16, _dtypes.quint16, ])
  min_input = _ops.convert_to_tensor(min_input, _dtypes.float32)
  max_input = _ops.convert_to_tensor(max_input, _dtypes.float32)
  _inputs_flat = [input, min_input, max_input]
  _attrs = ("T", _attr_T, "ksize", ksize, "strides", strides, "padding",
  padding)
  _result = _execute.execute(b"QuantizedMaxPool", 3, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "QuantizedMaxPool", _inputs_flat, _attrs, _result)
  _result = _QuantizedMaxPoolOutput._make(_result)
  return _result

_QuantizedReluOutput = collections.namedtuple(
    "QuantizedRelu",
    ["activations", "min_activations", "max_activations"])


TV_QuantizedRelu_Tinput = TypeVar("TV_QuantizedRelu_Tinput", _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8)
TV_QuantizedRelu_out_type = TypeVar("TV_QuantizedRelu_out_type", _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8)

def quantized_relu(features: Annotated[Any, TV_QuantizedRelu_Tinput], min_features: Annotated[Any, _atypes.Float32], max_features: Annotated[Any, _atypes.Float32], out_type:TV_QuantizedRelu_out_type=_dtypes.quint8, name=None):
  r"""Computes Quantized Rectified Linear: `max(features, 0)`

  Args:
    features: A `Tensor`. Must be one of the following types: `qint8`, `quint8`, `qint32`, `qint16`, `quint16`.
    min_features: A `Tensor` of type `float32`.
      The float value that the lowest quantized value represents.
    max_features: A `Tensor` of type `float32`.
      The float value that the highest quantized value represents.
    out_type: An optional `tf.DType` from: `tf.qint8, tf.quint8, tf.qint32, tf.qint16, tf.quint16`. Defaults to `tf.quint8`.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (activations, min_activations, max_activations).

    activations: A `Tensor` of type `out_type`.
    min_activations: A `Tensor` of type `float32`.
    max_activations: A `Tensor` of type `float32`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "QuantizedRelu", name, features, min_features, max_features,
        "out_type", out_type)
      _result = _QuantizedReluOutput._make(_result)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return quantized_relu_eager_fallback(
          features, min_features, max_features, out_type=out_type, name=name,
          ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if out_type is None:
    out_type = _dtypes.quint8
  out_type = _execute.make_type(out_type, "out_type")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "QuantizedRelu", features=features, min_features=min_features,
                         max_features=max_features, out_type=out_type,
                         name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("Tinput", _op._get_attr_type("Tinput"), "out_type",
              _op._get_attr_type("out_type"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "QuantizedRelu", _inputs_flat, _attrs, _result)
  _result = _QuantizedReluOutput._make(_result)
  return _result

QuantizedRelu = tf_export("raw_ops.QuantizedRelu")(_ops.to_raw_op(quantized_relu))


def quantized_relu_eager_fallback(features: Annotated[Any, TV_QuantizedRelu_Tinput], min_features: Annotated[Any, _atypes.Float32], max_features: Annotated[Any, _atypes.Float32], out_type: TV_QuantizedRelu_out_type, name, ctx):
  if out_type is None:
    out_type = _dtypes.quint8
  out_type = _execute.make_type(out_type, "out_type")
  _attr_Tinput, (features,) = _execute.args_to_matching_eager([features], ctx, [_dtypes.qint8, _dtypes.quint8, _dtypes.qint32, _dtypes.qint16, _dtypes.quint16, ])
  min_features = _ops.convert_to_tensor(min_features, _dtypes.float32)
  max_features = _ops.convert_to_tensor(max_features, _dtypes.float32)
  _inputs_flat = [features, min_features, max_features]
  _attrs = ("Tinput", _attr_Tinput, "out_type", out_type)
  _result = _execute.execute(b"QuantizedRelu", 3, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "QuantizedRelu", _inputs_flat, _attrs, _result)
  _result = _QuantizedReluOutput._make(_result)
  return _result

_QuantizedRelu6Output = collections.namedtuple(
    "QuantizedRelu6",
    ["activations", "min_activations", "max_activations"])


TV_QuantizedRelu6_Tinput = TypeVar("TV_QuantizedRelu6_Tinput", _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8)
TV_QuantizedRelu6_out_type = TypeVar("TV_QuantizedRelu6_out_type", _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8)

def quantized_relu6(features: Annotated[Any, TV_QuantizedRelu6_Tinput], min_features: Annotated[Any, _atypes.Float32], max_features: Annotated[Any, _atypes.Float32], out_type:TV_QuantizedRelu6_out_type=_dtypes.quint8, name=None):
  r"""Computes Quantized Rectified Linear 6: `min(max(features, 0), 6)`

  Args:
    features: A `Tensor`. Must be one of the following types: `qint8`, `quint8`, `qint32`, `qint16`, `quint16`.
    min_features: A `Tensor` of type `float32`.
      The float value that the lowest quantized value represents.
    max_features: A `Tensor` of type `float32`.
      The float value that the highest quantized value represents.
    out_type: An optional `tf.DType` from: `tf.qint8, tf.quint8, tf.qint32, tf.qint16, tf.quint16`. Defaults to `tf.quint8`.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (activations, min_activations, max_activations).

    activations: A `Tensor` of type `out_type`.
    min_activations: A `Tensor` of type `float32`.
    max_activations: A `Tensor` of type `float32`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "QuantizedRelu6", name, features, min_features, max_features,
        "out_type", out_type)
      _result = _QuantizedRelu6Output._make(_result)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return quantized_relu6_eager_fallback(
          features, min_features, max_features, out_type=out_type, name=name,
          ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if out_type is None:
    out_type = _dtypes.quint8
  out_type = _execute.make_type(out_type, "out_type")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "QuantizedRelu6", features=features, min_features=min_features,
                          max_features=max_features, out_type=out_type,
                          name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("Tinput", _op._get_attr_type("Tinput"), "out_type",
              _op._get_attr_type("out_type"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "QuantizedRelu6", _inputs_flat, _attrs, _result)
  _result = _QuantizedRelu6Output._make(_result)
  return _result

QuantizedRelu6 = tf_export("raw_ops.QuantizedRelu6")(_ops.to_raw_op(quantized_relu6))


def quantized_relu6_eager_fallback(features: Annotated[Any, TV_QuantizedRelu6_Tinput], min_features: Annotated[Any, _atypes.Float32], max_features: Annotated[Any, _atypes.Float32], out_type: TV_QuantizedRelu6_out_type, name, ctx):
  if out_type is None:
    out_type = _dtypes.quint8
  out_type = _execute.make_type(out_type, "out_type")
  _attr_Tinput, (features,) = _execute.args_to_matching_eager([features], ctx, [_dtypes.qint8, _dtypes.quint8, _dtypes.qint32, _dtypes.qint16, _dtypes.quint16, ])
  min_features = _ops.convert_to_tensor(min_features, _dtypes.float32)
  max_features = _ops.convert_to_tensor(max_features, _dtypes.float32)
  _inputs_flat = [features, min_features, max_features]
  _attrs = ("Tinput", _attr_Tinput, "out_type", out_type)
  _result = _execute.execute(b"QuantizedRelu6", 3, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "QuantizedRelu6", _inputs_flat, _attrs, _result)
  _result = _QuantizedRelu6Output._make(_result)
  return _result

_QuantizedReluXOutput = collections.namedtuple(
    "QuantizedReluX",
    ["activations", "min_activations", "max_activations"])


TV_QuantizedReluX_Tinput = TypeVar("TV_QuantizedReluX_Tinput", _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8)
TV_QuantizedReluX_out_type = TypeVar("TV_QuantizedReluX_out_type", _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8)

def quantized_relu_x(features: Annotated[Any, TV_QuantizedReluX_Tinput], max_value: Annotated[Any, _atypes.Float32], min_features: Annotated[Any, _atypes.Float32], max_features: Annotated[Any, _atypes.Float32], out_type:TV_QuantizedReluX_out_type=_dtypes.quint8, name=None):
  r"""Computes Quantized Rectified Linear X: `min(max(features, 0), max_value)`

  Args:
    features: A `Tensor`. Must be one of the following types: `qint8`, `quint8`, `qint32`, `qint16`, `quint16`.
    max_value: A `Tensor` of type `float32`.
    min_features: A `Tensor` of type `float32`.
      The float value that the lowest quantized value represents.
    max_features: A `Tensor` of type `float32`.
      The float value that the highest quantized value represents.
    out_type: An optional `tf.DType` from: `tf.qint8, tf.quint8, tf.qint32, tf.qint16, tf.quint16`. Defaults to `tf.quint8`.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (activations, min_activations, max_activations).

    activations: A `Tensor` of type `out_type`.
    min_activations: A `Tensor` of type `float32`.
    max_activations: A `Tensor` of type `float32`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "QuantizedReluX", name, features, max_value, min_features,
        max_features, "out_type", out_type)
      _result = _QuantizedReluXOutput._make(_result)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return quantized_relu_x_eager_fallback(
          features, max_value, min_features, max_features, out_type=out_type,
          name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if out_type is None:
    out_type = _dtypes.quint8
  out_type = _execute.make_type(out_type, "out_type")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "QuantizedReluX", features=features, max_value=max_value,
                          min_features=min_features,
                          max_features=max_features, out_type=out_type,
                          name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("Tinput", _op._get_attr_type("Tinput"), "out_type",
              _op._get_attr_type("out_type"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "QuantizedReluX", _inputs_flat, _attrs, _result)
  _result = _QuantizedReluXOutput._make(_result)
  return _result

QuantizedReluX = tf_export("raw_ops.QuantizedReluX")(_ops.to_raw_op(quantized_relu_x))


def quantized_relu_x_eager_fallback(features: Annotated[Any, TV_QuantizedReluX_Tinput], max_value: Annotated[Any, _atypes.Float32], min_features: Annotated[Any, _atypes.Float32], max_features: Annotated[Any, _atypes.Float32], out_type: TV_QuantizedReluX_out_type, name, ctx):
  if out_type is None:
    out_type = _dtypes.quint8
  out_type = _execute.make_type(out_type, "out_type")
  _attr_Tinput, (features,) = _execute.args_to_matching_eager([features], ctx, [_dtypes.qint8, _dtypes.quint8, _dtypes.qint32, _dtypes.qint16, _dtypes.quint16, ])
  max_value = _ops.convert_to_tensor(max_value, _dtypes.float32)
  min_features = _ops.convert_to_tensor(min_features, _dtypes.float32)
  max_features = _ops.convert_to_tensor(max_features, _dtypes.float32)
  _inputs_flat = [features, max_value, min_features, max_features]
  _attrs = ("Tinput", _attr_Tinput, "out_type", out_type)
  _result = _execute.execute(b"QuantizedReluX", 3, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "QuantizedReluX", _inputs_flat, _attrs, _result)
  _result = _QuantizedReluXOutput._make(_result)
  return _result


TV_Relu_T = TypeVar("TV_Relu_T", _atypes.BFloat16, _atypes.Float32, _atypes.Float64, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int64, _atypes.Int8, _atypes.QInt8, _atypes.UInt16, _atypes.UInt32, _atypes.UInt64, _atypes.UInt8)

@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('nn.relu')
def relu(features: Annotated[Any, TV_Relu_T], name=None) -> Annotated[Any, TV_Relu_T]:
  r"""Computes rectified linear: `max(features, 0)`.

  See: https://en.wikipedia.org/wiki/Rectifier_(neural_networks)
  Example usage:
  >>> tf.nn.relu([-2., 0., 3.]).numpy()
  array([0., 0., 3.], dtype=float32)

  Args:
    features: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`, `int64`, `bfloat16`, `uint16`, `half`, `uint32`, `uint64`, `qint8`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `features`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "Relu", name, features)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_relu(
          (features, name,), None)
      if _result is not NotImplemented:
        return _result
      return relu_eager_fallback(
          features, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            relu, (), dict(features=features, name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_relu(
        (features, name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "Relu", features=features, name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          relu, (), dict(features=features, name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "Relu", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

Relu = tf_export("raw_ops.Relu")(_ops.to_raw_op(relu))
_dispatcher_for_relu = relu._tf_type_based_dispatcher.Dispatch


def relu_eager_fallback(features: Annotated[Any, TV_Relu_T], name, ctx) -> Annotated[Any, TV_Relu_T]:
  _attr_T, (features,) = _execute.args_to_matching_eager([features], ctx, [_dtypes.float32, _dtypes.float64, _dtypes.int32, _dtypes.uint8, _dtypes.int16, _dtypes.int8, _dtypes.int64, _dtypes.bfloat16, _dtypes.uint16, _dtypes.half, _dtypes.uint32, _dtypes.uint64, _dtypes.qint8, ])
  _inputs_flat = [features]
  _attrs = ("T", _attr_T)
  _result = _execute.execute(b"Relu", 1, inputs=_inputs_flat, attrs=_attrs,
                             ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "Relu", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_Relu6_T = TypeVar("TV_Relu6_T", _atypes.BFloat16, _atypes.Float32, _atypes.Float64, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int64, _atypes.Int8, _atypes.UInt16, _atypes.UInt32, _atypes.UInt64, _atypes.UInt8)

def relu6(features: Annotated[Any, TV_Relu6_T], name=None) -> Annotated[Any, TV_Relu6_T]:
  r"""Computes rectified linear 6: `min(max(features, 0), 6)`.

  Args:
    features: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`, `int64`, `bfloat16`, `uint16`, `half`, `uint32`, `uint64`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `features`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "Relu6", name, features)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return relu6_eager_fallback(
          features, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "Relu6", features=features, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "Relu6", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

Relu6 = tf_export("raw_ops.Relu6")(_ops.to_raw_op(relu6))


def relu6_eager_fallback(features: Annotated[Any, TV_Relu6_T], name, ctx) -> Annotated[Any, TV_Relu6_T]:
  _attr_T, (features,) = _execute.args_to_matching_eager([features], ctx, [_dtypes.float32, _dtypes.float64, _dtypes.int32, _dtypes.uint8, _dtypes.int16, _dtypes.int8, _dtypes.int64, _dtypes.bfloat16, _dtypes.uint16, _dtypes.half, _dtypes.uint32, _dtypes.uint64, ])
  _inputs_flat = [features]
  _attrs = ("T", _attr_T)
  _result = _execute.execute(b"Relu6", 1, inputs=_inputs_flat, attrs=_attrs,
                             ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "Relu6", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_Relu6Grad_T = TypeVar("TV_Relu6Grad_T", _atypes.BFloat16, _atypes.Float32, _atypes.Float64, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int64, _atypes.Int8, _atypes.UInt16, _atypes.UInt32, _atypes.UInt64, _atypes.UInt8)

def relu6_grad(gradients: Annotated[Any, TV_Relu6Grad_T], features: Annotated[Any, TV_Relu6Grad_T], name=None) -> Annotated[Any, TV_Relu6Grad_T]:
  r"""Computes rectified linear 6 gradients for a Relu6 operation.

  Args:
    gradients: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`, `int64`, `bfloat16`, `uint16`, `half`, `uint32`, `uint64`.
      The backpropagated gradients to the corresponding Relu6 operation.
    features: A `Tensor`. Must have the same type as `gradients`.
      The features passed as input to the corresponding Relu6 operation, or
      its output; using either one produces the same result.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `gradients`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "Relu6Grad", name, gradients, features)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return relu6_grad_eager_fallback(
          gradients, features, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "Relu6Grad", gradients=gradients, features=features, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "Relu6Grad", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

Relu6Grad = tf_export("raw_ops.Relu6Grad")(_ops.to_raw_op(relu6_grad))


def relu6_grad_eager_fallback(gradients: Annotated[Any, TV_Relu6Grad_T], features: Annotated[Any, TV_Relu6Grad_T], name, ctx) -> Annotated[Any, TV_Relu6Grad_T]:
  _attr_T, _inputs_T = _execute.args_to_matching_eager([gradients, features], ctx, [_dtypes.float32, _dtypes.float64, _dtypes.int32, _dtypes.uint8, _dtypes.int16, _dtypes.int8, _dtypes.int64, _dtypes.bfloat16, _dtypes.uint16, _dtypes.half, _dtypes.uint32, _dtypes.uint64, ])
  (gradients, features) = _inputs_T
  _inputs_flat = [gradients, features]
  _attrs = ("T", _attr_T)
  _result = _execute.execute(b"Relu6Grad", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "Relu6Grad", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_ReluGrad_T = TypeVar("TV_ReluGrad_T", _atypes.BFloat16, _atypes.Float32, _atypes.Float64, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int64, _atypes.Int8, _atypes.UInt16, _atypes.UInt32, _atypes.UInt64, _atypes.UInt8)

def relu_grad(gradients: Annotated[Any, TV_ReluGrad_T], features: Annotated[Any, TV_ReluGrad_T], name=None) -> Annotated[Any, TV_ReluGrad_T]:
  r"""Computes rectified linear gradients for a Relu operation.

  Args:
    gradients: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`, `int64`, `bfloat16`, `uint16`, `half`, `uint32`, `uint64`.
      The backpropagated gradients to the corresponding Relu operation.
    features: A `Tensor`. Must have the same type as `gradients`.
      The features passed as input to the corresponding Relu operation, OR
      the outputs of that operation (both work equivalently).
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `gradients`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "ReluGrad", name, gradients, features)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return relu_grad_eager_fallback(
          gradients, features, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "ReluGrad", gradients=gradients, features=features, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "ReluGrad", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

ReluGrad = tf_export("raw_ops.ReluGrad")(_ops.to_raw_op(relu_grad))


def relu_grad_eager_fallback(gradients: Annotated[Any, TV_ReluGrad_T], features: Annotated[Any, TV_ReluGrad_T], name, ctx) -> Annotated[Any, TV_ReluGrad_T]:
  _attr_T, _inputs_T = _execute.args_to_matching_eager([gradients, features], ctx, [_dtypes.float32, _dtypes.float64, _dtypes.int32, _dtypes.uint8, _dtypes.int16, _dtypes.int8, _dtypes.int64, _dtypes.bfloat16, _dtypes.uint16, _dtypes.half, _dtypes.uint32, _dtypes.uint64, ])
  (gradients, features) = _inputs_T
  _inputs_flat = [gradients, features]
  _attrs = ("T", _attr_T)
  _result = _execute.execute(b"ReluGrad", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "ReluGrad", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_Selu_T = TypeVar("TV_Selu_T", _atypes.BFloat16, _atypes.Float32, _atypes.Float64, _atypes.Half)

@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('nn.selu')
def selu(features: Annotated[Any, TV_Selu_T], name=None) -> Annotated[Any, TV_Selu_T]:
  r"""Computes scaled exponential linear: `scale * alpha * (exp(features) - 1)`

  if < 0, `scale * features` otherwise.

  To be used together with
  `initializer = tf.variance_scaling_initializer(factor=1.0, mode='FAN_IN')`.
  For correct dropout, use `tf.contrib.nn.alpha_dropout`.

  See [Self-Normalizing Neural Networks](https://arxiv.org/abs/1706.02515)

  Args:
    features: A `Tensor`. Must be one of the following types: `half`, `bfloat16`, `float32`, `float64`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `features`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "Selu", name, features)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_selu(
          (features, name,), None)
      if _result is not NotImplemented:
        return _result
      return selu_eager_fallback(
          features, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            selu, (), dict(features=features, name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_selu(
        (features, name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "Selu", features=features, name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          selu, (), dict(features=features, name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "Selu", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

Selu = tf_export("raw_ops.Selu")(_ops.to_raw_op(selu))
_dispatcher_for_selu = selu._tf_type_based_dispatcher.Dispatch


def selu_eager_fallback(features: Annotated[Any, TV_Selu_T], name, ctx) -> Annotated[Any, TV_Selu_T]:
  _attr_T, (features,) = _execute.args_to_matching_eager([features], ctx, [_dtypes.half, _dtypes.bfloat16, _dtypes.float32, _dtypes.float64, ])
  _inputs_flat = [features]
  _attrs = ("T", _attr_T)
  _result = _execute.execute(b"Selu", 1, inputs=_inputs_flat, attrs=_attrs,
                             ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "Selu", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_SeluGrad_T = TypeVar("TV_SeluGrad_T", _atypes.BFloat16, _atypes.Float32, _atypes.Float64, _atypes.Half)

def selu_grad(gradients: Annotated[Any, TV_SeluGrad_T], outputs: Annotated[Any, TV_SeluGrad_T], name=None) -> Annotated[Any, TV_SeluGrad_T]:
  r"""Computes gradients for the scaled exponential linear (Selu) operation.

  Args:
    gradients: A `Tensor`. Must be one of the following types: `half`, `bfloat16`, `float32`, `float64`.
      The backpropagated gradients to the corresponding Selu operation.
    outputs: A `Tensor`. Must have the same type as `gradients`.
      The outputs of the corresponding Selu operation.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `gradients`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "SeluGrad", name, gradients, outputs)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return selu_grad_eager_fallback(
          gradients, outputs, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "SeluGrad", gradients=gradients, outputs=outputs, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "SeluGrad", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

SeluGrad = tf_export("raw_ops.SeluGrad")(_ops.to_raw_op(selu_grad))


def selu_grad_eager_fallback(gradients: Annotated[Any, TV_SeluGrad_T], outputs: Annotated[Any, TV_SeluGrad_T], name, ctx) -> Annotated[Any, TV_SeluGrad_T]:
  _attr_T, _inputs_T = _execute.args_to_matching_eager([gradients, outputs], ctx, [_dtypes.half, _dtypes.bfloat16, _dtypes.float32, _dtypes.float64, ])
  (gradients, outputs) = _inputs_T
  _inputs_flat = [gradients, outputs]
  _attrs = ("T", _attr_T)
  _result = _execute.execute(b"SeluGrad", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "SeluGrad", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_Softmax_T = TypeVar("TV_Softmax_T", _atypes.BFloat16, _atypes.Float32, _atypes.Float64, _atypes.Half)

def softmax(logits: Annotated[Any, TV_Softmax_T], name=None) -> Annotated[Any, TV_Softmax_T]:
  r"""Computes softmax activations.

  For each batch `i` and class `j` we have

      $$softmax[i, j] = exp(logits[i, j]) / sum_j(exp(logits[i, j]))$$

  Args:
    logits: A `Tensor`. Must be one of the following types: `half`, `bfloat16`, `float32`, `float64`.
      2-D with shape `[batch_size, num_classes]`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `logits`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "Softmax", name, logits)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return softmax_eager_fallback(
          logits, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "Softmax", logits=logits, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "Softmax", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

Softmax = tf_export("raw_ops.Softmax")(_ops.to_raw_op(softmax))


def softmax_eager_fallback(logits: Annotated[Any, TV_Softmax_T], name, ctx) -> Annotated[Any, TV_Softmax_T]:
  _attr_T, (logits,) = _execute.args_to_matching_eager([logits], ctx, [_dtypes.half, _dtypes.bfloat16, _dtypes.float32, _dtypes.float64, ])
  _inputs_flat = [logits]
  _attrs = ("T", _attr_T)
  _result = _execute.execute(b"Softmax", 1, inputs=_inputs_flat, attrs=_attrs,
                             ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "Softmax", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

_SoftmaxCrossEntropyWithLogitsOutput = collections.namedtuple(
    "SoftmaxCrossEntropyWithLogits",
    ["loss", "backprop"])


TV_SoftmaxCrossEntropyWithLogits_T = TypeVar("TV_SoftmaxCrossEntropyWithLogits_T", _atypes.BFloat16, _atypes.Float32, _atypes.Float64, _atypes.Half)

def softmax_cross_entropy_with_logits(features: Annotated[Any, TV_SoftmaxCrossEntropyWithLogits_T], labels: Annotated[Any, TV_SoftmaxCrossEntropyWithLogits_T], name=None):
  r"""Computes softmax cross entropy cost and gradients to backpropagate.

  Inputs are the logits, not probabilities.

  Args:
    features: A `Tensor`. Must be one of the following types: `half`, `bfloat16`, `float32`, `float64`.
      batch_size x num_classes matrix
    labels: A `Tensor`. Must have the same type as `features`.
      batch_size x num_classes matrix
      The caller must ensure that each batch of labels represents a valid
      probability distribution.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (loss, backprop).

    loss: A `Tensor`. Has the same type as `features`.
    backprop: A `Tensor`. Has the same type as `features`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "SoftmaxCrossEntropyWithLogits", name, features, labels)
      _result = _SoftmaxCrossEntropyWithLogitsOutput._make(_result)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return softmax_cross_entropy_with_logits_eager_fallback(
          features, labels, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "SoftmaxCrossEntropyWithLogits", features=features, labels=labels,
                                         name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "SoftmaxCrossEntropyWithLogits", _inputs_flat, _attrs, _result)
  _result = _SoftmaxCrossEntropyWithLogitsOutput._make(_result)
  return _result

SoftmaxCrossEntropyWithLogits = tf_export("raw_ops.SoftmaxCrossEntropyWithLogits")(_ops.to_raw_op(softmax_cross_entropy_with_logits))


def softmax_cross_entropy_with_logits_eager_fallback(features: Annotated[Any, TV_SoftmaxCrossEntropyWithLogits_T], labels: Annotated[Any, TV_SoftmaxCrossEntropyWithLogits_T], name, ctx):
  _attr_T, _inputs_T = _execute.args_to_matching_eager([features, labels], ctx, [_dtypes.half, _dtypes.bfloat16, _dtypes.float32, _dtypes.float64, ])
  (features, labels) = _inputs_T
  _inputs_flat = [features, labels]
  _attrs = ("T", _attr_T)
  _result = _execute.execute(b"SoftmaxCrossEntropyWithLogits", 2,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "SoftmaxCrossEntropyWithLogits", _inputs_flat, _attrs, _result)
  _result = _SoftmaxCrossEntropyWithLogitsOutput._make(_result)
  return _result


TV_Softplus_T = TypeVar("TV_Softplus_T", _atypes.BFloat16, _atypes.Float32, _atypes.Float64, _atypes.Half)

def softplus(features: Annotated[Any, TV_Softplus_T], name=None) -> Annotated[Any, TV_Softplus_T]:
  r"""TODO: add doc.

  Args:
    features: A `Tensor`. Must be one of the following types: `half`, `bfloat16`, `float32`, `float64`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `features`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "Softplus", name, features)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return softplus_eager_fallback(
          features, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "Softplus", features=features, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "Softplus", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

Softplus = tf_export("raw_ops.Softplus")(_ops.to_raw_op(softplus))


def softplus_eager_fallback(features: Annotated[Any, TV_Softplus_T], name, ctx) -> Annotated[Any, TV_Softplus_T]:
  _attr_T, (features,) = _execute.args_to_matching_eager([features], ctx, [_dtypes.half, _dtypes.bfloat16, _dtypes.float32, _dtypes.float64, ])
  _inputs_flat = [features]
  _attrs = ("T", _attr_T)
  _result = _execute.execute(b"Softplus", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "Softplus", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_SoftplusGrad_T = TypeVar("TV_SoftplusGrad_T", _atypes.BFloat16, _atypes.Float32, _atypes.Float64, _atypes.Half)

def softplus_grad(gradients: Annotated[Any, TV_SoftplusGrad_T], features: Annotated[Any, TV_SoftplusGrad_T], name=None) -> Annotated[Any, TV_SoftplusGrad_T]:
  r"""Computes softplus gradients for a softplus operation.

  Args:
    gradients: A `Tensor`. Must be one of the following types: `half`, `bfloat16`, `float32`, `float64`.
      The backpropagated gradients to the corresponding softplus operation.
    features: A `Tensor`. Must have the same type as `gradients`.
      The features passed as input to the corresponding softplus operation.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `gradients`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "SoftplusGrad", name, gradients, features)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return softplus_grad_eager_fallback(
          gradients, features, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "SoftplusGrad", gradients=gradients, features=features, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "SoftplusGrad", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

SoftplusGrad = tf_export("raw_ops.SoftplusGrad")(_ops.to_raw_op(softplus_grad))


def softplus_grad_eager_fallback(gradients: Annotated[Any, TV_SoftplusGrad_T], features: Annotated[Any, TV_SoftplusGrad_T], name, ctx) -> Annotated[Any, TV_SoftplusGrad_T]:
  _attr_T, _inputs_T = _execute.args_to_matching_eager([gradients, features], ctx, [_dtypes.half, _dtypes.bfloat16, _dtypes.float32, _dtypes.float64, ])
  (gradients, features) = _inputs_T
  _inputs_flat = [gradients, features]
  _attrs = ("T", _attr_T)
  _result = _execute.execute(b"SoftplusGrad", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "SoftplusGrad", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_Softsign_T = TypeVar("TV_Softsign_T", _atypes.BFloat16, _atypes.Float32, _atypes.Float64, _atypes.Half)

@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('nn.softsign', 'math.softsign')
def softsign(features: Annotated[Any, TV_Softsign_T], name=None) -> Annotated[Any, TV_Softsign_T]:
  r"""Computes softsign: `features / (abs(features) + 1)`.

  Args:
    features: A `Tensor`. Must be one of the following types: `half`, `bfloat16`, `float32`, `float64`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `features`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "Softsign", name, features)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_softsign(
          (features, name,), None)
      if _result is not NotImplemented:
        return _result
      return softsign_eager_fallback(
          features, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            softsign, (), dict(features=features, name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_softsign(
        (features, name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "Softsign", features=features, name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          softsign, (), dict(features=features, name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "Softsign", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

Softsign = tf_export("raw_ops.Softsign")(_ops.to_raw_op(softsign))
_dispatcher_for_softsign = softsign._tf_type_based_dispatcher.Dispatch


def softsign_eager_fallback(features: Annotated[Any, TV_Softsign_T], name, ctx) -> Annotated[Any, TV_Softsign_T]:
  _attr_T, (features,) = _execute.args_to_matching_eager([features], ctx, [_dtypes.half, _dtypes.bfloat16, _dtypes.float32, _dtypes.float64, ])
  _inputs_flat = [features]
  _attrs = ("T", _attr_T)
  _result = _execute.execute(b"Softsign", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "Softsign", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_SoftsignGrad_T = TypeVar("TV_SoftsignGrad_T", _atypes.BFloat16, _atypes.Float32, _atypes.Float64, _atypes.Half)

def softsign_grad(gradients: Annotated[Any, TV_SoftsignGrad_T], features: Annotated[Any, TV_SoftsignGrad_T], name=None) -> Annotated[Any, TV_SoftsignGrad_T]:
  r"""Computes softsign gradients for a softsign operation.

  Args:
    gradients: A `Tensor`. Must be one of the following types: `half`, `bfloat16`, `float32`, `float64`.
      The backpropagated gradients to the corresponding softsign operation.
    features: A `Tensor`. Must have the same type as `gradients`.
      The features passed as input to the corresponding softsign operation.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `gradients`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "SoftsignGrad", name, gradients, features)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return softsign_grad_eager_fallback(
          gradients, features, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "SoftsignGrad", gradients=gradients, features=features, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "SoftsignGrad", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

SoftsignGrad = tf_export("raw_ops.SoftsignGrad")(_ops.to_raw_op(softsign_grad))


def softsign_grad_eager_fallback(gradients: Annotated[Any, TV_SoftsignGrad_T], features: Annotated[Any, TV_SoftsignGrad_T], name, ctx) -> Annotated[Any, TV_SoftsignGrad_T]:
  _attr_T, _inputs_T = _execute.args_to_matching_eager([gradients, features], ctx, [_dtypes.half, _dtypes.bfloat16, _dtypes.float32, _dtypes.float64, ])
  (gradients, features) = _inputs_T
  _inputs_flat = [gradients, features]
  _attrs = ("T", _attr_T)
  _result = _execute.execute(b"SoftsignGrad", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "SoftsignGrad", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

_SparseSoftmaxCrossEntropyWithLogitsOutput = collections.namedtuple(
    "SparseSoftmaxCrossEntropyWithLogits",
    ["loss", "backprop"])


TV_SparseSoftmaxCrossEntropyWithLogits_T = TypeVar("TV_SparseSoftmaxCrossEntropyWithLogits_T", _atypes.BFloat16, _atypes.Float32, _atypes.Float64, _atypes.Half)
TV_SparseSoftmaxCrossEntropyWithLogits_Tlabels = TypeVar("TV_SparseSoftmaxCrossEntropyWithLogits_Tlabels", _atypes.Int32, _atypes.Int64)

def sparse_softmax_cross_entropy_with_logits(features: Annotated[Any, TV_SparseSoftmaxCrossEntropyWithLogits_T], labels: Annotated[Any, TV_SparseSoftmaxCrossEntropyWithLogits_Tlabels], name=None):
  r"""Computes softmax cross entropy cost and gradients to backpropagate.

  Unlike `SoftmaxCrossEntropyWithLogits`, this operation does not accept
  a matrix of label probabilities, but rather a single label per row
  of features.  This label is considered to have probability 1.0 for the
  given row.

  Inputs are the logits, not probabilities.

  Args:
    features: A `Tensor`. Must be one of the following types: `half`, `bfloat16`, `float32`, `float64`.
      batch_size x num_classes matrix
    labels: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      batch_size vector with values in [0, num_classes).
      This is the label for the given minibatch entry.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (loss, backprop).

    loss: A `Tensor`. Has the same type as `features`.
    backprop: A `Tensor`. Has the same type as `features`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "SparseSoftmaxCrossEntropyWithLogits", name, features, labels)
      _result = _SparseSoftmaxCrossEntropyWithLogitsOutput._make(_result)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return sparse_softmax_cross_entropy_with_logits_eager_fallback(
          features, labels, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "SparseSoftmaxCrossEntropyWithLogits", features=features,
                                               labels=labels, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"), "Tlabels",
              _op._get_attr_type("Tlabels"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "SparseSoftmaxCrossEntropyWithLogits", _inputs_flat, _attrs, _result)
  _result = _SparseSoftmaxCrossEntropyWithLogitsOutput._make(_result)
  return _result

SparseSoftmaxCrossEntropyWithLogits = tf_export("raw_ops.SparseSoftmaxCrossEntropyWithLogits")(_ops.to_raw_op(sparse_softmax_cross_entropy_with_logits))


def sparse_softmax_cross_entropy_with_logits_eager_fallback(features: Annotated[Any, TV_SparseSoftmaxCrossEntropyWithLogits_T], labels: Annotated[Any, TV_SparseSoftmaxCrossEntropyWithLogits_Tlabels], name, ctx):
  _attr_T, (features,) = _execute.args_to_matching_eager([features], ctx, [_dtypes.half, _dtypes.bfloat16, _dtypes.float32, _dtypes.float64, ])
  _attr_Tlabels, (labels,) = _execute.args_to_matching_eager([labels], ctx, [_dtypes.int32, _dtypes.int64, ], _dtypes.int64)
  _inputs_flat = [features, labels]
  _attrs = ("T", _attr_T, "Tlabels", _attr_Tlabels)
  _result = _execute.execute(b"SparseSoftmaxCrossEntropyWithLogits", 2,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "SparseSoftmaxCrossEntropyWithLogits", _inputs_flat, _attrs, _result)
  _result = _SparseSoftmaxCrossEntropyWithLogitsOutput._make(_result)
  return _result

_TopKOutput = collections.namedtuple(
    "TopK",
    ["values", "indices"])


TV_TopK_T = TypeVar("TV_TopK_T", _atypes.BFloat16, _atypes.Float32, _atypes.Float64, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int64, _atypes.Int8, _atypes.UInt16, _atypes.UInt32, _atypes.UInt64, _atypes.UInt8)

def top_k(input: Annotated[Any, TV_TopK_T], k: int, sorted:bool=True, name=None):
  r"""Finds values and indices of the `k` largest elements for the last dimension.

  If the input is a vector (rank-1), finds the `k` largest entries in the vector
  and outputs their values and indices as vectors.  Thus `values[j]` is the
  `j`-th largest entry in `input`, and its index is `indices[j]`.

  For matrices (resp. higher rank input), computes the top `k` entries in each
  row (resp. vector along the last dimension).  Thus,

      values.shape = indices.shape = input.shape[:-1] + [k]

  If two elements are equal, the lower-index element appears first.

  If `k` varies dynamically, use `TopKV2` below.

  Args:
    input: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`, `int64`, `bfloat16`, `uint16`, `half`, `uint32`, `uint64`.
      1-D or higher with last dimension at least `k`.
    k: An `int` that is `>= 0`.
      Number of top elements to look for along the last dimension (along each
      row for matrices).
    sorted: An optional `bool`. Defaults to `True`.
      If true the resulting `k` elements will be sorted by the values in
      descending order.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (values, indices).

    values: A `Tensor`. Has the same type as `input`.
    indices: A `Tensor` of type `int32`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "TopK", name, input, "k", k, "sorted", sorted)
      _result = _TopKOutput._make(_result)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return top_k_eager_fallback(
          input, k=k, sorted=sorted, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  k = _execute.make_int(k, "k")
  if sorted is None:
    sorted = True
  sorted = _execute.make_bool(sorted, "sorted")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "TopK", input=input, k=k, sorted=sorted, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("k", _op._get_attr_int("k"), "sorted",
              _op._get_attr_bool("sorted"), "T", _op._get_attr_type("T"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "TopK", _inputs_flat, _attrs, _result)
  _result = _TopKOutput._make(_result)
  return _result

TopK = tf_export("raw_ops.TopK")(_ops.to_raw_op(top_k))


def top_k_eager_fallback(input: Annotated[Any, TV_TopK_T], k: int, sorted: bool, name, ctx):
  k = _execute.make_int(k, "k")
  if sorted is None:
    sorted = True
  sorted = _execute.make_bool(sorted, "sorted")
  _attr_T, (input,) = _execute.args_to_matching_eager([input], ctx, [_dtypes.float32, _dtypes.float64, _dtypes.int32, _dtypes.uint8, _dtypes.int16, _dtypes.int8, _dtypes.int64, _dtypes.bfloat16, _dtypes.uint16, _dtypes.half, _dtypes.uint32, _dtypes.uint64, ])
  _inputs_flat = [input]
  _attrs = ("k", k, "sorted", sorted, "T", _attr_T)
  _result = _execute.execute(b"TopK", 2, inputs=_inputs_flat, attrs=_attrs,
                             ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "TopK", _inputs_flat, _attrs, _result)
  _result = _TopKOutput._make(_result)
  return _result

_TopKV2Output = collections.namedtuple(
    "TopKV2",
    ["values", "indices"])


TV_TopKV2_T = TypeVar("TV_TopKV2_T", _atypes.BFloat16, _atypes.Float32, _atypes.Float64, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int64, _atypes.Int8, _atypes.UInt16, _atypes.UInt32, _atypes.UInt64, _atypes.UInt8)
TV_TopKV2_Tk = TypeVar("TV_TopKV2_Tk", _atypes.Int16, _atypes.Int32, _atypes.Int64)
TV_TopKV2_index_type = TypeVar("TV_TopKV2_index_type", _atypes.Int16, _atypes.Int32, _atypes.Int64)

def top_kv2(input: Annotated[Any, TV_TopKV2_T], k: Annotated[Any, TV_TopKV2_Tk], sorted:bool=True, index_type:TV_TopKV2_index_type=_dtypes.int32, name=None):
  r"""Finds values and indices of the `k` largest elements for the last dimension.

  If the input is a vector (rank-1), finds the `k` largest entries in the vector
  and outputs their values and indices as vectors.  Thus `values[j]` is the
  `j`-th largest entry in `input`, and its index is `indices[j]`.

  For matrices (resp. higher rank input), computes the top `k` entries in each
  row (resp. vector along the last dimension).  Thus,

      values.shape = indices.shape = input.shape[:-1] + [k]

  If two elements are equal, the lower-index element appears first.

  Args:
    input: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`, `int64`, `bfloat16`, `uint16`, `half`, `uint32`, `uint64`.
      1-D or higher with last dimension at least `k`.
    k: A `Tensor`. Must be one of the following types: `int16`, `int32`, `int64`.
      0-D.  Number of top elements to look for along the last dimension (along each
      row for matrices).
    sorted: An optional `bool`. Defaults to `True`.
      If true the resulting `k` elements will be sorted by the values in
      descending order.
    index_type: An optional `tf.DType` from: `tf.int16, tf.int32, tf.int64`. Defaults to `tf.int32`.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (values, indices).

    values: A `Tensor`. Has the same type as `input`.
    indices: A `Tensor` of type `index_type`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "TopKV2", name, input, k, "sorted", sorted, "index_type",
        index_type)
      _result = _TopKV2Output._make(_result)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return top_kv2_eager_fallback(
          input, k, sorted=sorted, index_type=index_type, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if sorted is None:
    sorted = True
  sorted = _execute.make_bool(sorted, "sorted")
  if index_type is None:
    index_type = _dtypes.int32
  index_type = _execute.make_type(index_type, "index_type")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "TopKV2", input=input, k=k, sorted=sorted, index_type=index_type,
                  name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("sorted", _op._get_attr_bool("sorted"), "T",
              _op._get_attr_type("T"), "Tk", _op._get_attr_type("Tk"),
              "index_type", _op._get_attr_type("index_type"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "TopKV2", _inputs_flat, _attrs, _result)
  _result = _TopKV2Output._make(_result)
  return _result

TopKV2 = tf_export("raw_ops.TopKV2")(_ops.to_raw_op(top_kv2))


def top_kv2_eager_fallback(input: Annotated[Any, TV_TopKV2_T], k: Annotated[Any, TV_TopKV2_Tk], sorted: bool, index_type: TV_TopKV2_index_type, name, ctx):
  if sorted is None:
    sorted = True
  sorted = _execute.make_bool(sorted, "sorted")
  if index_type is None:
    index_type = _dtypes.int32
  index_type = _execute.make_type(index_type, "index_type")
  _attr_T, (input,) = _execute.args_to_matching_eager([input], ctx, [_dtypes.float32, _dtypes.float64, _dtypes.int32, _dtypes.uint8, _dtypes.int16, _dtypes.int8, _dtypes.int64, _dtypes.bfloat16, _dtypes.uint16, _dtypes.half, _dtypes.uint32, _dtypes.uint64, ])
  _attr_Tk, (k,) = _execute.args_to_matching_eager([k], ctx, [_dtypes.int16, _dtypes.int32, _dtypes.int64, ], _dtypes.int32)
  _inputs_flat = [input, k]
  _attrs = ("sorted", sorted, "T", _attr_T, "Tk", _attr_Tk, "index_type",
  index_type)
  _result = _execute.execute(b"TopKV2", 2, inputs=_inputs_flat, attrs=_attrs,
                             ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "TopKV2", _inputs_flat, _attrs, _result)
  _result = _TopKV2Output._make(_result)
  return _result

