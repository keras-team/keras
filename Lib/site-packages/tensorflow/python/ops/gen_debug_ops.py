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

TV_Copy_T = TypeVar("TV_Copy_T", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float16, _atypes.Float32, _atypes.Float64, _atypes.Float8e4m3fn, _atypes.Float8e5m2, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int4, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.Resource, _atypes.String, _atypes.UInt16, _atypes.UInt32, _atypes.UInt4, _atypes.UInt64, _atypes.UInt8, _atypes.Variant)

def copy(input: Annotated[Any, TV_Copy_T], tensor_name:str="", debug_ops_spec=[], name=None) -> Annotated[Any, TV_Copy_T]:
  r"""Copy a tensor from CPU-to-CPU or GPU-to-GPU.

  Performs CPU-to-CPU or GPU-to-GPU deep-copying of tensor, depending on the
  device on which the tensor is allocated.
  N.B.: If the all downstream attached debug ops are disabled given the current
  gRPC gating status, the output will simply forward the input tensor without
  deep-copying. See the documentation of Debug* ops for more details.

  Unlike the CopyHost Op, this op does not have HostMemory constraint on its
  input or output.

  Args:
    input: A `Tensor`. Input tensor.
    tensor_name: An optional `string`. Defaults to `""`.
      The name of the input tensor.
    debug_ops_spec: An optional list of `strings`. Defaults to `[]`.
      A list of debug op spec (op, url, gated_grpc) for attached debug
      ops. Each element of the list has the format
      <debug_op>;<grpc_url>;<gated_grpc>, wherein gated_grpc is boolean represented
      as 0/1. E.g., "DebugIdentity;grpc://foo:3333;1",
      "DebugIdentity;file:///tmp/tfdbg_1;0".
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "Copy", name, input, "tensor_name", tensor_name,
        "debug_ops_spec", debug_ops_spec)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return copy_eager_fallback(
          input, tensor_name=tensor_name, debug_ops_spec=debug_ops_spec,
          name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if tensor_name is None:
    tensor_name = ""
  tensor_name = _execute.make_str(tensor_name, "tensor_name")
  if debug_ops_spec is None:
    debug_ops_spec = []
  if not isinstance(debug_ops_spec, (list, tuple)):
    raise TypeError(
        "Expected list for 'debug_ops_spec' argument to "
        "'copy' Op, not %r." % debug_ops_spec)
  debug_ops_spec = [_execute.make_str(_s, "debug_ops_spec") for _s in debug_ops_spec]
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "Copy", input=input, tensor_name=tensor_name,
                debug_ops_spec=debug_ops_spec, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"), "tensor_name",
              _op.get_attr("tensor_name"), "debug_ops_spec",
              _op.get_attr("debug_ops_spec"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "Copy", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

Copy = tf_export("raw_ops.Copy")(_ops.to_raw_op(copy))


def copy_eager_fallback(input: Annotated[Any, TV_Copy_T], tensor_name: str, debug_ops_spec, name, ctx) -> Annotated[Any, TV_Copy_T]:
  if tensor_name is None:
    tensor_name = ""
  tensor_name = _execute.make_str(tensor_name, "tensor_name")
  if debug_ops_spec is None:
    debug_ops_spec = []
  if not isinstance(debug_ops_spec, (list, tuple)):
    raise TypeError(
        "Expected list for 'debug_ops_spec' argument to "
        "'copy' Op, not %r." % debug_ops_spec)
  debug_ops_spec = [_execute.make_str(_s, "debug_ops_spec") for _s in debug_ops_spec]
  _attr_T, (input,) = _execute.args_to_matching_eager([input], ctx, [])
  _inputs_flat = [input]
  _attrs = ("T", _attr_T, "tensor_name", tensor_name, "debug_ops_spec",
  debug_ops_spec)
  _result = _execute.execute(b"Copy", 1, inputs=_inputs_flat, attrs=_attrs,
                             ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "Copy", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_CopyHost_T = TypeVar("TV_CopyHost_T", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float16, _atypes.Float32, _atypes.Float64, _atypes.Float8e4m3fn, _atypes.Float8e5m2, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int4, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.Resource, _atypes.String, _atypes.UInt16, _atypes.UInt32, _atypes.UInt4, _atypes.UInt64, _atypes.UInt8, _atypes.Variant)

def copy_host(input: Annotated[Any, TV_CopyHost_T], tensor_name:str="", debug_ops_spec=[], name=None) -> Annotated[Any, TV_CopyHost_T]:
  r"""Copy a tensor to host.

  Performs CPU-to-CPU deep-copying of tensor.
  N.B.: If the all downstream attached debug ops are disabled given the current
  gRPC gating status, the output will simply forward the input tensor without
  deep-copying. See the documentation of Debug* ops for more details.

  Unlike the Copy Op, this op has HostMemory constraint on its input or output.

  Args:
    input: A `Tensor`. Input tensor.
    tensor_name: An optional `string`. Defaults to `""`.
      The name of the input tensor.
    debug_ops_spec: An optional list of `strings`. Defaults to `[]`.
      A list of debug op spec (op, url, gated_grpc) for attached debug
      ops. Each element of the list has the format
      <debug_op>;<grpc_url>;<gated_grpc>, wherein gated_grpc is boolean represented
      as 0/1. E.g., "DebugIdentity;grpc://foo:3333;1",
      "DebugIdentity;file:///tmp/tfdbg_1;0".
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "CopyHost", name, input, "tensor_name", tensor_name,
        "debug_ops_spec", debug_ops_spec)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return copy_host_eager_fallback(
          input, tensor_name=tensor_name, debug_ops_spec=debug_ops_spec,
          name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if tensor_name is None:
    tensor_name = ""
  tensor_name = _execute.make_str(tensor_name, "tensor_name")
  if debug_ops_spec is None:
    debug_ops_spec = []
  if not isinstance(debug_ops_spec, (list, tuple)):
    raise TypeError(
        "Expected list for 'debug_ops_spec' argument to "
        "'copy_host' Op, not %r." % debug_ops_spec)
  debug_ops_spec = [_execute.make_str(_s, "debug_ops_spec") for _s in debug_ops_spec]
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "CopyHost", input=input, tensor_name=tensor_name,
                    debug_ops_spec=debug_ops_spec, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"), "tensor_name",
              _op.get_attr("tensor_name"), "debug_ops_spec",
              _op.get_attr("debug_ops_spec"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "CopyHost", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

CopyHost = tf_export("raw_ops.CopyHost")(_ops.to_raw_op(copy_host))


def copy_host_eager_fallback(input: Annotated[Any, TV_CopyHost_T], tensor_name: str, debug_ops_spec, name, ctx) -> Annotated[Any, TV_CopyHost_T]:
  if tensor_name is None:
    tensor_name = ""
  tensor_name = _execute.make_str(tensor_name, "tensor_name")
  if debug_ops_spec is None:
    debug_ops_spec = []
  if not isinstance(debug_ops_spec, (list, tuple)):
    raise TypeError(
        "Expected list for 'debug_ops_spec' argument to "
        "'copy_host' Op, not %r." % debug_ops_spec)
  debug_ops_spec = [_execute.make_str(_s, "debug_ops_spec") for _s in debug_ops_spec]
  _attr_T, (input,) = _execute.args_to_matching_eager([input], ctx, [])
  _inputs_flat = [input]
  _attrs = ("T", _attr_T, "tensor_name", tensor_name, "debug_ops_spec",
  debug_ops_spec)
  _result = _execute.execute(b"CopyHost", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "CopyHost", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_DebugIdentity_T = TypeVar("TV_DebugIdentity_T", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float16, _atypes.Float32, _atypes.Float64, _atypes.Float8e4m3fn, _atypes.Float8e5m2, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int4, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.Resource, _atypes.String, _atypes.UInt16, _atypes.UInt32, _atypes.UInt4, _atypes.UInt64, _atypes.UInt8, _atypes.Variant)

def debug_identity(input: Annotated[Any, TV_DebugIdentity_T], device_name:str="", tensor_name:str="", debug_urls=[], gated_grpc:bool=False, name=None) -> Annotated[Any, TV_DebugIdentity_T]:
  r"""Provides an identity mapping of the non-Ref type input tensor for debugging.

  Provides an identity mapping of the non-Ref type input tensor for debugging.

  Args:
    input: A `Tensor`. Input tensor, non-Reference type
    device_name: An optional `string`. Defaults to `""`.
      Name of the device on which the tensor resides.
    tensor_name: An optional `string`. Defaults to `""`.
      Name of the input tensor.
    debug_urls: An optional list of `strings`. Defaults to `[]`.
      List of URLs to debug targets, e.g.,
        file:///foo/tfdbg_dump, grpc:://localhost:11011
    gated_grpc: An optional `bool`. Defaults to `False`.
      Whether this op will be gated. If any of the debug_urls of this
        debug node is of the grpc:// scheme, when the value of this attribute is set
        to True, the data will not actually be sent via the grpc stream unless this
        debug op has been enabled at the debug_url. If all of the debug_urls of this
        debug node are of the grpc:// scheme and the debug op is enabled at none of
        them, the output will be an empty Tensor.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "DebugIdentity", name, input, "device_name", device_name,
        "tensor_name", tensor_name, "debug_urls", debug_urls, "gated_grpc",
        gated_grpc)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return debug_identity_eager_fallback(
          input, device_name=device_name, tensor_name=tensor_name,
          debug_urls=debug_urls, gated_grpc=gated_grpc, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if device_name is None:
    device_name = ""
  device_name = _execute.make_str(device_name, "device_name")
  if tensor_name is None:
    tensor_name = ""
  tensor_name = _execute.make_str(tensor_name, "tensor_name")
  if debug_urls is None:
    debug_urls = []
  if not isinstance(debug_urls, (list, tuple)):
    raise TypeError(
        "Expected list for 'debug_urls' argument to "
        "'debug_identity' Op, not %r." % debug_urls)
  debug_urls = [_execute.make_str(_s, "debug_urls") for _s in debug_urls]
  if gated_grpc is None:
    gated_grpc = False
  gated_grpc = _execute.make_bool(gated_grpc, "gated_grpc")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "DebugIdentity", input=input, device_name=device_name,
                         tensor_name=tensor_name, debug_urls=debug_urls,
                         gated_grpc=gated_grpc, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"), "device_name",
              _op.get_attr("device_name"), "tensor_name",
              _op.get_attr("tensor_name"), "debug_urls",
              _op.get_attr("debug_urls"), "gated_grpc",
              _op._get_attr_bool("gated_grpc"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "DebugIdentity", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

DebugIdentity = tf_export("raw_ops.DebugIdentity")(_ops.to_raw_op(debug_identity))


def debug_identity_eager_fallback(input: Annotated[Any, TV_DebugIdentity_T], device_name: str, tensor_name: str, debug_urls, gated_grpc: bool, name, ctx) -> Annotated[Any, TV_DebugIdentity_T]:
  if device_name is None:
    device_name = ""
  device_name = _execute.make_str(device_name, "device_name")
  if tensor_name is None:
    tensor_name = ""
  tensor_name = _execute.make_str(tensor_name, "tensor_name")
  if debug_urls is None:
    debug_urls = []
  if not isinstance(debug_urls, (list, tuple)):
    raise TypeError(
        "Expected list for 'debug_urls' argument to "
        "'debug_identity' Op, not %r." % debug_urls)
  debug_urls = [_execute.make_str(_s, "debug_urls") for _s in debug_urls]
  if gated_grpc is None:
    gated_grpc = False
  gated_grpc = _execute.make_bool(gated_grpc, "gated_grpc")
  _attr_T, (input,) = _execute.args_to_matching_eager([input], ctx, [])
  _inputs_flat = [input]
  _attrs = ("T", _attr_T, "device_name", device_name, "tensor_name",
  tensor_name, "debug_urls", debug_urls, "gated_grpc", gated_grpc)
  _result = _execute.execute(b"DebugIdentity", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "DebugIdentity", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_DebugIdentityV2_T = TypeVar("TV_DebugIdentityV2_T", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float16, _atypes.Float32, _atypes.Float64, _atypes.Float8e4m3fn, _atypes.Float8e5m2, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int4, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.Resource, _atypes.String, _atypes.UInt16, _atypes.UInt32, _atypes.UInt4, _atypes.UInt64, _atypes.UInt8, _atypes.Variant)

def debug_identity_v2(input: Annotated[Any, TV_DebugIdentityV2_T], tfdbg_context_id:str="", op_name:str="", output_slot:int=-1, tensor_debug_mode:int=-1, debug_urls=[], circular_buffer_size:int=1000, tfdbg_run_id:str="", name=None) -> Annotated[Any, TV_DebugIdentityV2_T]:
  r"""Debug Identity V2 Op.

  Provides an identity mapping from input to output, while writing the content of
  the input tensor by calling DebugEventsWriter.

  The semantics of the input tensor depends on tensor_debug_mode. In typical
  usage, the input tensor comes directly from the user computation only when
  graph_debug_mode is FULL_TENSOR (see protobuf/debug_event.proto for a
  list of all the possible values of graph_debug_mode). For the other debug modes,
  the input tensor should be produced by an additional op or subgraph that
  computes summary information about one or more tensors.

  Args:
    input: A `Tensor`. Input tensor, non-Reference type
    tfdbg_context_id: An optional `string`. Defaults to `""`.
      A tfdbg-generated ID for the context that the op belongs to,
        e.g., a concrete compiled tf.function.
    op_name: An optional `string`. Defaults to `""`.
      Optional. Name of the op that the debug op is concerned with.
        Used only for single-tensor trace.
    output_slot: An optional `int`. Defaults to `-1`.
      Optional. Output slot index of the tensor that the debug op
        is concerned with. Used only for single-tensor trace.
    tensor_debug_mode: An optional `int`. Defaults to `-1`.
      TensorDebugMode enum value. See debug_event.proto for details.
    debug_urls: An optional list of `strings`. Defaults to `[]`.
      List of URLs to debug targets, e.g., file:///foo/tfdbg_dump.
    circular_buffer_size: An optional `int`. Defaults to `1000`.
    tfdbg_run_id: An optional `string`. Defaults to `""`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "DebugIdentityV2", name, input, "tfdbg_context_id",
        tfdbg_context_id, "op_name", op_name, "output_slot", output_slot,
        "tensor_debug_mode", tensor_debug_mode, "debug_urls", debug_urls,
        "circular_buffer_size", circular_buffer_size, "tfdbg_run_id",
        tfdbg_run_id)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return debug_identity_v2_eager_fallback(
          input, tfdbg_context_id=tfdbg_context_id, op_name=op_name,
          output_slot=output_slot, tensor_debug_mode=tensor_debug_mode,
          debug_urls=debug_urls, circular_buffer_size=circular_buffer_size,
          tfdbg_run_id=tfdbg_run_id, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if tfdbg_context_id is None:
    tfdbg_context_id = ""
  tfdbg_context_id = _execute.make_str(tfdbg_context_id, "tfdbg_context_id")
  if op_name is None:
    op_name = ""
  op_name = _execute.make_str(op_name, "op_name")
  if output_slot is None:
    output_slot = -1
  output_slot = _execute.make_int(output_slot, "output_slot")
  if tensor_debug_mode is None:
    tensor_debug_mode = -1
  tensor_debug_mode = _execute.make_int(tensor_debug_mode, "tensor_debug_mode")
  if debug_urls is None:
    debug_urls = []
  if not isinstance(debug_urls, (list, tuple)):
    raise TypeError(
        "Expected list for 'debug_urls' argument to "
        "'debug_identity_v2' Op, not %r." % debug_urls)
  debug_urls = [_execute.make_str(_s, "debug_urls") for _s in debug_urls]
  if circular_buffer_size is None:
    circular_buffer_size = 1000
  circular_buffer_size = _execute.make_int(circular_buffer_size, "circular_buffer_size")
  if tfdbg_run_id is None:
    tfdbg_run_id = ""
  tfdbg_run_id = _execute.make_str(tfdbg_run_id, "tfdbg_run_id")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "DebugIdentityV2", input=input, tfdbg_context_id=tfdbg_context_id,
                           op_name=op_name, output_slot=output_slot,
                           tensor_debug_mode=tensor_debug_mode,
                           debug_urls=debug_urls,
                           circular_buffer_size=circular_buffer_size,
                           tfdbg_run_id=tfdbg_run_id, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"), "tfdbg_context_id",
              _op.get_attr("tfdbg_context_id"), "op_name",
              _op.get_attr("op_name"), "output_slot",
              _op._get_attr_int("output_slot"), "tensor_debug_mode",
              _op._get_attr_int("tensor_debug_mode"), "debug_urls",
              _op.get_attr("debug_urls"), "circular_buffer_size",
              _op._get_attr_int("circular_buffer_size"), "tfdbg_run_id",
              _op.get_attr("tfdbg_run_id"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "DebugIdentityV2", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

DebugIdentityV2 = tf_export("raw_ops.DebugIdentityV2")(_ops.to_raw_op(debug_identity_v2))


def debug_identity_v2_eager_fallback(input: Annotated[Any, TV_DebugIdentityV2_T], tfdbg_context_id: str, op_name: str, output_slot: int, tensor_debug_mode: int, debug_urls, circular_buffer_size: int, tfdbg_run_id: str, name, ctx) -> Annotated[Any, TV_DebugIdentityV2_T]:
  if tfdbg_context_id is None:
    tfdbg_context_id = ""
  tfdbg_context_id = _execute.make_str(tfdbg_context_id, "tfdbg_context_id")
  if op_name is None:
    op_name = ""
  op_name = _execute.make_str(op_name, "op_name")
  if output_slot is None:
    output_slot = -1
  output_slot = _execute.make_int(output_slot, "output_slot")
  if tensor_debug_mode is None:
    tensor_debug_mode = -1
  tensor_debug_mode = _execute.make_int(tensor_debug_mode, "tensor_debug_mode")
  if debug_urls is None:
    debug_urls = []
  if not isinstance(debug_urls, (list, tuple)):
    raise TypeError(
        "Expected list for 'debug_urls' argument to "
        "'debug_identity_v2' Op, not %r." % debug_urls)
  debug_urls = [_execute.make_str(_s, "debug_urls") for _s in debug_urls]
  if circular_buffer_size is None:
    circular_buffer_size = 1000
  circular_buffer_size = _execute.make_int(circular_buffer_size, "circular_buffer_size")
  if tfdbg_run_id is None:
    tfdbg_run_id = ""
  tfdbg_run_id = _execute.make_str(tfdbg_run_id, "tfdbg_run_id")
  _attr_T, (input,) = _execute.args_to_matching_eager([input], ctx, [])
  _inputs_flat = [input]
  _attrs = ("T", _attr_T, "tfdbg_context_id", tfdbg_context_id, "op_name",
  op_name, "output_slot", output_slot, "tensor_debug_mode", tensor_debug_mode,
  "debug_urls", debug_urls, "circular_buffer_size", circular_buffer_size,
  "tfdbg_run_id", tfdbg_run_id)
  _result = _execute.execute(b"DebugIdentityV2", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "DebugIdentityV2", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_DebugIdentityV3_T = TypeVar("TV_DebugIdentityV3_T", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float16, _atypes.Float32, _atypes.Float64, _atypes.Float8e4m3fn, _atypes.Float8e5m2, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int4, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.Resource, _atypes.String, _atypes.UInt16, _atypes.UInt32, _atypes.UInt4, _atypes.UInt64, _atypes.UInt8, _atypes.Variant)

def debug_identity_v3(input: Annotated[Any, TV_DebugIdentityV3_T], device_name:str="", tensor_name:str="", io_of_node:str="", is_input:bool=False, io_index:int=-1, debug_urls=[], gated_grpc:bool=False, name=None) -> Annotated[Any, TV_DebugIdentityV3_T]:
  r"""Provides an identity mapping of the non-Ref type input tensor for debugging.

  Provides an identity mapping of the non-Ref type input tensor for debugging.

  Args:
    input: A `Tensor`. Input tensor, non-Reference type
    device_name: An optional `string`. Defaults to `""`.
      Name of the device on which the tensor resides.
    tensor_name: An optional `string`. Defaults to `""`.
      Name of the input tensor.
    io_of_node: An optional `string`. Defaults to `""`.
      Name of the node of which the tensor is an input or output.
    is_input: An optional `bool`. Defaults to `False`.
      If true, the tensor is an input of the node; otherwise the output.
    io_index: An optional `int`. Defaults to `-1`.
      The index of which the tensor is an input or output of the node.
    debug_urls: An optional list of `strings`. Defaults to `[]`.
      List of URLs to debug targets, e.g.,
        file:///foo/tfdbg_dump, grpc:://localhost:11011
    gated_grpc: An optional `bool`. Defaults to `False`.
      Whether this op will be gated. If any of the debug_urls of this
        debug node is of the grpc:// scheme, when the value of this attribute is set
        to True, the data will not actually be sent via the grpc stream unless this
        debug op has been enabled at the debug_url. If all of the debug_urls of this
        debug node are of the grpc:// scheme and the debug op is enabled at none of
        them, the output will be an empty Tensor.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "DebugIdentityV3", name, input, "device_name", device_name,
        "tensor_name", tensor_name, "io_of_node", io_of_node, "is_input",
        is_input, "io_index", io_index, "debug_urls", debug_urls,
        "gated_grpc", gated_grpc)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return debug_identity_v3_eager_fallback(
          input, device_name=device_name, tensor_name=tensor_name,
          io_of_node=io_of_node, is_input=is_input, io_index=io_index,
          debug_urls=debug_urls, gated_grpc=gated_grpc, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if device_name is None:
    device_name = ""
  device_name = _execute.make_str(device_name, "device_name")
  if tensor_name is None:
    tensor_name = ""
  tensor_name = _execute.make_str(tensor_name, "tensor_name")
  if io_of_node is None:
    io_of_node = ""
  io_of_node = _execute.make_str(io_of_node, "io_of_node")
  if is_input is None:
    is_input = False
  is_input = _execute.make_bool(is_input, "is_input")
  if io_index is None:
    io_index = -1
  io_index = _execute.make_int(io_index, "io_index")
  if debug_urls is None:
    debug_urls = []
  if not isinstance(debug_urls, (list, tuple)):
    raise TypeError(
        "Expected list for 'debug_urls' argument to "
        "'debug_identity_v3' Op, not %r." % debug_urls)
  debug_urls = [_execute.make_str(_s, "debug_urls") for _s in debug_urls]
  if gated_grpc is None:
    gated_grpc = False
  gated_grpc = _execute.make_bool(gated_grpc, "gated_grpc")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "DebugIdentityV3", input=input, device_name=device_name,
                           tensor_name=tensor_name, io_of_node=io_of_node,
                           is_input=is_input, io_index=io_index,
                           debug_urls=debug_urls, gated_grpc=gated_grpc,
                           name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"), "device_name",
              _op.get_attr("device_name"), "tensor_name",
              _op.get_attr("tensor_name"), "io_of_node",
              _op.get_attr("io_of_node"), "is_input",
              _op._get_attr_bool("is_input"), "io_index",
              _op._get_attr_int("io_index"), "debug_urls",
              _op.get_attr("debug_urls"), "gated_grpc",
              _op._get_attr_bool("gated_grpc"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "DebugIdentityV3", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

DebugIdentityV3 = tf_export("raw_ops.DebugIdentityV3")(_ops.to_raw_op(debug_identity_v3))


def debug_identity_v3_eager_fallback(input: Annotated[Any, TV_DebugIdentityV3_T], device_name: str, tensor_name: str, io_of_node: str, is_input: bool, io_index: int, debug_urls, gated_grpc: bool, name, ctx) -> Annotated[Any, TV_DebugIdentityV3_T]:
  if device_name is None:
    device_name = ""
  device_name = _execute.make_str(device_name, "device_name")
  if tensor_name is None:
    tensor_name = ""
  tensor_name = _execute.make_str(tensor_name, "tensor_name")
  if io_of_node is None:
    io_of_node = ""
  io_of_node = _execute.make_str(io_of_node, "io_of_node")
  if is_input is None:
    is_input = False
  is_input = _execute.make_bool(is_input, "is_input")
  if io_index is None:
    io_index = -1
  io_index = _execute.make_int(io_index, "io_index")
  if debug_urls is None:
    debug_urls = []
  if not isinstance(debug_urls, (list, tuple)):
    raise TypeError(
        "Expected list for 'debug_urls' argument to "
        "'debug_identity_v3' Op, not %r." % debug_urls)
  debug_urls = [_execute.make_str(_s, "debug_urls") for _s in debug_urls]
  if gated_grpc is None:
    gated_grpc = False
  gated_grpc = _execute.make_bool(gated_grpc, "gated_grpc")
  _attr_T, (input,) = _execute.args_to_matching_eager([input], ctx, [])
  _inputs_flat = [input]
  _attrs = ("T", _attr_T, "device_name", device_name, "tensor_name",
  tensor_name, "io_of_node", io_of_node, "is_input", is_input, "io_index",
  io_index, "debug_urls", debug_urls, "gated_grpc", gated_grpc)
  _result = _execute.execute(b"DebugIdentityV3", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "DebugIdentityV3", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_DebugNanCount_T = TypeVar("TV_DebugNanCount_T", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float16, _atypes.Float32, _atypes.Float64, _atypes.Float8e4m3fn, _atypes.Float8e5m2, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int4, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.Resource, _atypes.String, _atypes.UInt16, _atypes.UInt32, _atypes.UInt4, _atypes.UInt64, _atypes.UInt8, _atypes.Variant)

def debug_nan_count(input: Annotated[Any, TV_DebugNanCount_T], device_name:str="", tensor_name:str="", debug_urls=[], gated_grpc:bool=False, name=None) -> Annotated[Any, _atypes.Int64]:
  r"""Debug NaN Value Counter Op.

  Counts number of NaNs in the input tensor, for debugging.

  Args:
    input: A `Tensor`. Input tensor, non-Reference type.
    device_name: An optional `string`. Defaults to `""`.
    tensor_name: An optional `string`. Defaults to `""`.
      Name of the input tensor.
    debug_urls: An optional list of `strings`. Defaults to `[]`.
      List of URLs to debug targets, e.g.,
        file:///foo/tfdbg_dump, grpc:://localhost:11011.
    gated_grpc: An optional `bool`. Defaults to `False`.
       Whether this op will be gated. If any of the debug_urls of this
        debug node is of the grpc:// scheme, when the value of this attribute is set
        to True, the data will not actually be sent via the grpc stream unless this
        debug op has been enabled at the debug_url. If all of the debug_urls of this
        debug node are of the grpc:// scheme and the debug op is enabled at none of
        them, the output will be an empty Tensor.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `int64`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "DebugNanCount", name, input, "device_name", device_name,
        "tensor_name", tensor_name, "debug_urls", debug_urls, "gated_grpc",
        gated_grpc)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return debug_nan_count_eager_fallback(
          input, device_name=device_name, tensor_name=tensor_name,
          debug_urls=debug_urls, gated_grpc=gated_grpc, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if device_name is None:
    device_name = ""
  device_name = _execute.make_str(device_name, "device_name")
  if tensor_name is None:
    tensor_name = ""
  tensor_name = _execute.make_str(tensor_name, "tensor_name")
  if debug_urls is None:
    debug_urls = []
  if not isinstance(debug_urls, (list, tuple)):
    raise TypeError(
        "Expected list for 'debug_urls' argument to "
        "'debug_nan_count' Op, not %r." % debug_urls)
  debug_urls = [_execute.make_str(_s, "debug_urls") for _s in debug_urls]
  if gated_grpc is None:
    gated_grpc = False
  gated_grpc = _execute.make_bool(gated_grpc, "gated_grpc")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "DebugNanCount", input=input, device_name=device_name,
                         tensor_name=tensor_name, debug_urls=debug_urls,
                         gated_grpc=gated_grpc, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"), "device_name",
              _op.get_attr("device_name"), "tensor_name",
              _op.get_attr("tensor_name"), "debug_urls",
              _op.get_attr("debug_urls"), "gated_grpc",
              _op._get_attr_bool("gated_grpc"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "DebugNanCount", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

DebugNanCount = tf_export("raw_ops.DebugNanCount")(_ops.to_raw_op(debug_nan_count))


def debug_nan_count_eager_fallback(input: Annotated[Any, TV_DebugNanCount_T], device_name: str, tensor_name: str, debug_urls, gated_grpc: bool, name, ctx) -> Annotated[Any, _atypes.Int64]:
  if device_name is None:
    device_name = ""
  device_name = _execute.make_str(device_name, "device_name")
  if tensor_name is None:
    tensor_name = ""
  tensor_name = _execute.make_str(tensor_name, "tensor_name")
  if debug_urls is None:
    debug_urls = []
  if not isinstance(debug_urls, (list, tuple)):
    raise TypeError(
        "Expected list for 'debug_urls' argument to "
        "'debug_nan_count' Op, not %r." % debug_urls)
  debug_urls = [_execute.make_str(_s, "debug_urls") for _s in debug_urls]
  if gated_grpc is None:
    gated_grpc = False
  gated_grpc = _execute.make_bool(gated_grpc, "gated_grpc")
  _attr_T, (input,) = _execute.args_to_matching_eager([input], ctx, [])
  _inputs_flat = [input]
  _attrs = ("T", _attr_T, "device_name", device_name, "tensor_name",
  tensor_name, "debug_urls", debug_urls, "gated_grpc", gated_grpc)
  _result = _execute.execute(b"DebugNanCount", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "DebugNanCount", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_DebugNumericSummary_T = TypeVar("TV_DebugNumericSummary_T", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float16, _atypes.Float32, _atypes.Float64, _atypes.Float8e4m3fn, _atypes.Float8e5m2, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int4, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.Resource, _atypes.String, _atypes.UInt16, _atypes.UInt32, _atypes.UInt4, _atypes.UInt64, _atypes.UInt8, _atypes.Variant)

def debug_numeric_summary(input: Annotated[Any, TV_DebugNumericSummary_T], device_name:str="", tensor_name:str="", debug_urls=[], lower_bound:float=float('-inf'), upper_bound:float=float('inf'), mute_if_healthy:bool=False, gated_grpc:bool=False, name=None) -> Annotated[Any, _atypes.Float64]:
  r"""Debug Numeric Summary Op.

  Provide a basic summary of numeric value types, range and distribution.

  output: A double tensor of shape [14 + nDimensions], where nDimensions is the
    number of dimensions of the tensor's shape. The elements of output are:
    [0]: is initialized (1.0) or not (0.0).
    [1]: total number of elements
    [2]: NaN element count
    [3]: generalized -inf count: elements <= lower_bound. lower_bound is -inf by
      default.
    [4]: negative element count (excluding -inf), if lower_bound is the default
      -inf. Otherwise, this is the count of elements > lower_bound and < 0.
    [5]: zero element count
    [6]: positive element count (excluding +inf), if upper_bound is the default
      +inf. Otherwise, this is the count of elements < upper_bound and > 0.
    [7]: generalized +inf count, elements >= upper_bound. upper_bound is +inf by
      default.
  Output elements [1:8] are all zero, if the tensor is uninitialized.
    [8]: minimum of all non-inf and non-NaN elements.
         If uninitialized or no such element exists: +inf.
    [9]: maximum of all non-inf and non-NaN elements.
         If uninitialized or no such element exists: -inf.
    [10]: mean of all non-inf and non-NaN elements.
          If uninitialized or no such element exists: NaN.
    [11]: variance of all non-inf and non-NaN elements.
          If uninitialized or no such element exists: NaN.
    [12]: Data type of the tensor encoded as an enum integer. See the DataType
          proto for more details.
    [13]: Number of dimensions of the tensor (ndims).
    [14+]: Sizes of the dimensions.

  Args:
    input: A `Tensor`. Input tensor, non-Reference type.
    device_name: An optional `string`. Defaults to `""`.
    tensor_name: An optional `string`. Defaults to `""`.
      Name of the input tensor.
    debug_urls: An optional list of `strings`. Defaults to `[]`.
      List of URLs to debug targets, e.g.,
        file:///foo/tfdbg_dump, grpc:://localhost:11011.
    lower_bound: An optional `float`. Defaults to `float('-inf')`.
      (float) The lower bound <= which values will be included in the
        generalized -inf count. Default: -inf.
    upper_bound: An optional `float`. Defaults to `float('inf')`.
      (float) The upper bound >= which values will be included in the
        generalized +inf count. Default: +inf.
    mute_if_healthy: An optional `bool`. Defaults to `False`.
      (bool) Do not send data to the debug URLs unless at least one
        of elements [2], [3] and [7] (i.e., the nan count and the generalized -inf and
        inf counts) is non-zero.
    gated_grpc: An optional `bool`. Defaults to `False`.
      Whether this op will be gated. If any of the debug_urls of this
        debug node is of the grpc:// scheme, when the value of this attribute is set
        to True, the data will not actually be sent via the grpc stream unless this
        debug op has been enabled at the debug_url. If all of the debug_urls of this
        debug node are of the grpc:// scheme and the debug op is enabled at none of
        them, the output will be an empty Tensor.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `float64`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "DebugNumericSummary", name, input, "device_name", device_name,
        "tensor_name", tensor_name, "debug_urls", debug_urls, "lower_bound",
        lower_bound, "upper_bound", upper_bound, "mute_if_healthy",
        mute_if_healthy, "gated_grpc", gated_grpc)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return debug_numeric_summary_eager_fallback(
          input, device_name=device_name, tensor_name=tensor_name,
          debug_urls=debug_urls, lower_bound=lower_bound,
          upper_bound=upper_bound, mute_if_healthy=mute_if_healthy,
          gated_grpc=gated_grpc, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if device_name is None:
    device_name = ""
  device_name = _execute.make_str(device_name, "device_name")
  if tensor_name is None:
    tensor_name = ""
  tensor_name = _execute.make_str(tensor_name, "tensor_name")
  if debug_urls is None:
    debug_urls = []
  if not isinstance(debug_urls, (list, tuple)):
    raise TypeError(
        "Expected list for 'debug_urls' argument to "
        "'debug_numeric_summary' Op, not %r." % debug_urls)
  debug_urls = [_execute.make_str(_s, "debug_urls") for _s in debug_urls]
  if lower_bound is None:
    lower_bound = float('-inf')
  lower_bound = _execute.make_float(lower_bound, "lower_bound")
  if upper_bound is None:
    upper_bound = float('inf')
  upper_bound = _execute.make_float(upper_bound, "upper_bound")
  if mute_if_healthy is None:
    mute_if_healthy = False
  mute_if_healthy = _execute.make_bool(mute_if_healthy, "mute_if_healthy")
  if gated_grpc is None:
    gated_grpc = False
  gated_grpc = _execute.make_bool(gated_grpc, "gated_grpc")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "DebugNumericSummary", input=input, device_name=device_name,
                               tensor_name=tensor_name, debug_urls=debug_urls,
                               lower_bound=lower_bound,
                               upper_bound=upper_bound,
                               mute_if_healthy=mute_if_healthy,
                               gated_grpc=gated_grpc, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"), "device_name",
              _op.get_attr("device_name"), "tensor_name",
              _op.get_attr("tensor_name"), "debug_urls",
              _op.get_attr("debug_urls"), "lower_bound",
              _op.get_attr("lower_bound"), "upper_bound",
              _op.get_attr("upper_bound"), "mute_if_healthy",
              _op._get_attr_bool("mute_if_healthy"), "gated_grpc",
              _op._get_attr_bool("gated_grpc"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "DebugNumericSummary", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

DebugNumericSummary = tf_export("raw_ops.DebugNumericSummary")(_ops.to_raw_op(debug_numeric_summary))


def debug_numeric_summary_eager_fallback(input: Annotated[Any, TV_DebugNumericSummary_T], device_name: str, tensor_name: str, debug_urls, lower_bound: float, upper_bound: float, mute_if_healthy: bool, gated_grpc: bool, name, ctx) -> Annotated[Any, _atypes.Float64]:
  if device_name is None:
    device_name = ""
  device_name = _execute.make_str(device_name, "device_name")
  if tensor_name is None:
    tensor_name = ""
  tensor_name = _execute.make_str(tensor_name, "tensor_name")
  if debug_urls is None:
    debug_urls = []
  if not isinstance(debug_urls, (list, tuple)):
    raise TypeError(
        "Expected list for 'debug_urls' argument to "
        "'debug_numeric_summary' Op, not %r." % debug_urls)
  debug_urls = [_execute.make_str(_s, "debug_urls") for _s in debug_urls]
  if lower_bound is None:
    lower_bound = float('-inf')
  lower_bound = _execute.make_float(lower_bound, "lower_bound")
  if upper_bound is None:
    upper_bound = float('inf')
  upper_bound = _execute.make_float(upper_bound, "upper_bound")
  if mute_if_healthy is None:
    mute_if_healthy = False
  mute_if_healthy = _execute.make_bool(mute_if_healthy, "mute_if_healthy")
  if gated_grpc is None:
    gated_grpc = False
  gated_grpc = _execute.make_bool(gated_grpc, "gated_grpc")
  _attr_T, (input,) = _execute.args_to_matching_eager([input], ctx, [])
  _inputs_flat = [input]
  _attrs = ("T", _attr_T, "device_name", device_name, "tensor_name",
  tensor_name, "debug_urls", debug_urls, "lower_bound", lower_bound,
  "upper_bound", upper_bound, "mute_if_healthy", mute_if_healthy,
  "gated_grpc", gated_grpc)
  _result = _execute.execute(b"DebugNumericSummary", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "DebugNumericSummary", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_DebugNumericSummaryV2_output_dtype = TypeVar("TV_DebugNumericSummaryV2_output_dtype", _atypes.Float32, _atypes.Float64)
TV_DebugNumericSummaryV2_T = TypeVar("TV_DebugNumericSummaryV2_T", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float16, _atypes.Float32, _atypes.Float64, _atypes.Float8e4m3fn, _atypes.Float8e5m2, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int4, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.Resource, _atypes.String, _atypes.UInt16, _atypes.UInt32, _atypes.UInt4, _atypes.UInt64, _atypes.UInt8, _atypes.Variant)

def debug_numeric_summary_v2(input: Annotated[Any, TV_DebugNumericSummaryV2_T], output_dtype:TV_DebugNumericSummaryV2_output_dtype=_dtypes.float32, tensor_debug_mode:int=-1, tensor_id:int=-1, name=None) -> Annotated[Any, TV_DebugNumericSummaryV2_output_dtype]:
  r"""Debug Numeric Summary V2 Op.

  Computes a numeric summary of the input tensor. The shape of the output
  depends on the tensor_debug_mode attribute.
  This op is used internally by TensorFlow Debugger (tfdbg) v2.

  Args:
    input: A `Tensor`. Input tensor, to be summarized by the op.
    output_dtype: An optional `tf.DType` from: `tf.float32, tf.float64`. Defaults to `tf.float32`.
      Optional. The type of the output. Can be float32 or float64 (default: float32).
    tensor_debug_mode: An optional `int`. Defaults to `-1`.
      Tensor debug mode: the mode in which the input tensor is summarized
        by the op. See the TensorDebugMode enum in
        tensorflow/core/protobuf/debug_event.proto for details.

      Supported values:
        2 (CURT_HEALTH): Output a float32/64 tensor of shape [2]. The 1st
        element is the tensor_id, if provided, and -1 otherwise. The 2nd
        element is a bit which is set to 1 if the input tensor has an
        infinity or nan value, or zero otherwise.

        3 (CONCISE_HEALTH): Output a float32/64 tensor of shape [5]. The 1st
        element is the tensor_id, if provided, and -1 otherwise. The
        remaining four slots are the total number of elements, -infs,
        +infs, and nans in the input tensor respectively.

        4 (FULL_HEALTH): Output a float32/64 tensor of shape [11]. The 1st
        element is the tensor_id, if provided, and -1 otherwise. The 2nd
        element is the device_id, if provided, and -1 otherwise. The 3rd
        element holds the datatype value of the input tensor as according
        to the enumerated type in tensorflow/core/framework/types.proto.
        The remaining elements hold the total number of elements, -infs,
        +infs, nans, negative finite numbers, zeros, and positive finite
        numbers in the input tensor respectively.

        5 (SHAPE): Output a float32/64 tensor of shape [10]. The 1st
        element is the tensor_id, if provided, and -1 otherwise. The 2nd
        element holds the datatype value of the input tensor as according
        to the enumerated type in tensorflow/core/framework/types.proto.
        The 3rd element holds the rank of the tensor. The 4th element holds
        the number of elements within the tensor. Finally the remaining 6
        elements hold the shape of the tensor. If the rank of the tensor
        is lower than 6, the shape is right padded with zeros. If the rank
        is greater than 6, the head of the shape is truncated.

        6 (FULL_NUMERICS): Output a float32/64 tensor of shape [22]. The 1st
        element is the tensor_id, if provided, and -1 otherwise. The 2nd
        element is the device_id, if provided, and -1 otherwise. The 3rd
        element holds the datatype value of the input tensor as according
        to the enumerated type in tensorflow/core/framework/types.proto.
        The 4th element holds the rank of the tensor. The 5th to 11th
        elements hold the shape of the tensor. If the rank of the tensor
        is lower than 6, the shape is right padded with zeros. If the rank
        is greater than 6, the head of the shape is truncated. The 12th to
        18th elements hold the number of elements, -infs, +infs, nans,
        denormal floats, negative finite numbers, zeros, and positive
        finite numbers in the input tensor respectively. The final four
        elements hold the min value, max value, mean, and variance of the
        input tensor.

        8 (REDUCE_INF_NAN_THREE_SLOTS): Output a float32/64 tensor of shape
        [3]. The 1st element is -inf if any elements of the input tensor
        is -inf, or zero otherwise. The 2nd element is +inf if any elements
        of the input tensor is +inf, or zero otherwise.  The 3rd element is
        nan if any element of the input tensor is nan, or zero otherwise.
    tensor_id: An optional `int`. Defaults to `-1`.
      Optional. An integer identifier for the tensor being summarized by this op.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `output_dtype`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "DebugNumericSummaryV2", name, input, "output_dtype",
        output_dtype, "tensor_debug_mode", tensor_debug_mode, "tensor_id",
        tensor_id)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return debug_numeric_summary_v2_eager_fallback(
          input, output_dtype=output_dtype,
          tensor_debug_mode=tensor_debug_mode, tensor_id=tensor_id, name=name,
          ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if output_dtype is None:
    output_dtype = _dtypes.float32
  output_dtype = _execute.make_type(output_dtype, "output_dtype")
  if tensor_debug_mode is None:
    tensor_debug_mode = -1
  tensor_debug_mode = _execute.make_int(tensor_debug_mode, "tensor_debug_mode")
  if tensor_id is None:
    tensor_id = -1
  tensor_id = _execute.make_int(tensor_id, "tensor_id")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "DebugNumericSummaryV2", input=input, output_dtype=output_dtype,
                                 tensor_debug_mode=tensor_debug_mode,
                                 tensor_id=tensor_id, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("output_dtype", _op._get_attr_type("output_dtype"), "T",
              _op._get_attr_type("T"), "tensor_debug_mode",
              _op._get_attr_int("tensor_debug_mode"), "tensor_id",
              _op._get_attr_int("tensor_id"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "DebugNumericSummaryV2", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

DebugNumericSummaryV2 = tf_export("raw_ops.DebugNumericSummaryV2")(_ops.to_raw_op(debug_numeric_summary_v2))


def debug_numeric_summary_v2_eager_fallback(input: Annotated[Any, TV_DebugNumericSummaryV2_T], output_dtype: TV_DebugNumericSummaryV2_output_dtype, tensor_debug_mode: int, tensor_id: int, name, ctx) -> Annotated[Any, TV_DebugNumericSummaryV2_output_dtype]:
  if output_dtype is None:
    output_dtype = _dtypes.float32
  output_dtype = _execute.make_type(output_dtype, "output_dtype")
  if tensor_debug_mode is None:
    tensor_debug_mode = -1
  tensor_debug_mode = _execute.make_int(tensor_debug_mode, "tensor_debug_mode")
  if tensor_id is None:
    tensor_id = -1
  tensor_id = _execute.make_int(tensor_id, "tensor_id")
  _attr_T, (input,) = _execute.args_to_matching_eager([input], ctx, [])
  _inputs_flat = [input]
  _attrs = ("output_dtype", output_dtype, "T", _attr_T, "tensor_debug_mode",
  tensor_debug_mode, "tensor_id", tensor_id)
  _result = _execute.execute(b"DebugNumericSummaryV2", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "DebugNumericSummaryV2", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

