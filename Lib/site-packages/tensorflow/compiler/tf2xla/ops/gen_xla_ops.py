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

TV_XlaAllReduce_T = TypeVar("TV_XlaAllReduce_T", _atypes.BFloat16, _atypes.Float32, _atypes.Half, _atypes.Int32, _atypes.UInt32)

@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('xla_all_reduce')
def xla_all_reduce(input: Annotated[Any, TV_XlaAllReduce_T], group_assignment: Annotated[Any, _atypes.Int32], reduce_op: str, mode: str, name=None) -> Annotated[Any, TV_XlaAllReduce_T]:
  r"""Wraps the XLA AllReduce operator

    documented at https://www.tensorflow.org/xla/operation_semantics#allreduce.

  Args:
    input: A `Tensor`. Must be one of the following types: `half`, `bfloat16`, `float32`, `int32`, `uint32`.
      Array or a non-empty tuple of arrays to reduce across replicas.
    group_assignment: A `Tensor` of type `int32`.
      Groups between which the reductions are performed.
    reduce_op: A `string` from: `"Min", "Max", "Mul", "Add", "Mean"`.
      Reduction computation.
    mode: A `string` from: `"CrossReplica", "CrossReplicaAndPartition"`.
      group mode.
      CrossReplica: group_assignment contains replica_id. Each group contains the
        replicas for the current partition.
      CrossReplicaAndPartition: group_assignment contains replica_id. Each group
        contains the replicas for all partitions.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "XlaAllReduce", name, input, group_assignment, "reduce_op",
        reduce_op, "mode", mode)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_xla_all_reduce(
          (input, group_assignment, reduce_op, mode, name,), None)
      if _result is not NotImplemented:
        return _result
      return xla_all_reduce_eager_fallback(
          input, group_assignment, reduce_op=reduce_op, mode=mode, name=name,
          ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            xla_all_reduce, (), dict(input=input,
                                     group_assignment=group_assignment,
                                     reduce_op=reduce_op, mode=mode,
                                     name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_xla_all_reduce(
        (input, group_assignment, reduce_op, mode, name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  reduce_op = _execute.make_str(reduce_op, "reduce_op")
  mode = _execute.make_str(mode, "mode")
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "XlaAllReduce", input=input, group_assignment=group_assignment,
                        reduce_op=reduce_op, mode=mode, name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          xla_all_reduce, (), dict(input=input,
                                   group_assignment=group_assignment,
                                   reduce_op=reduce_op, mode=mode, name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"), "reduce_op",
              _op.get_attr("reduce_op"), "mode", _op.get_attr("mode"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "XlaAllReduce", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

XlaAllReduce = tf_export("raw_ops.XlaAllReduce")(_ops.to_raw_op(xla_all_reduce))
_dispatcher_for_xla_all_reduce = xla_all_reduce._tf_type_based_dispatcher.Dispatch


def xla_all_reduce_eager_fallback(input: Annotated[Any, TV_XlaAllReduce_T], group_assignment: Annotated[Any, _atypes.Int32], reduce_op: str, mode: str, name, ctx) -> Annotated[Any, TV_XlaAllReduce_T]:
  reduce_op = _execute.make_str(reduce_op, "reduce_op")
  mode = _execute.make_str(mode, "mode")
  _attr_T, (input,) = _execute.args_to_matching_eager([input], ctx, [_dtypes.half, _dtypes.bfloat16, _dtypes.float32, _dtypes.int32, _dtypes.uint32, ])
  group_assignment = _ops.convert_to_tensor(group_assignment, _dtypes.int32)
  _inputs_flat = [input, group_assignment]
  _attrs = ("T", _attr_T, "reduce_op", reduce_op, "mode", mode)
  _result = _execute.execute(b"XlaAllReduce", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "XlaAllReduce", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

_XlaBroadcastHelperOutput = collections.namedtuple(
    "XlaBroadcastHelper",
    ["lhs_output", "rhs_output"])


TV_XlaBroadcastHelper_T = TypeVar("TV_XlaBroadcastHelper_T", _atypes.BFloat16, _atypes.Complex128, _atypes.Complex64, _atypes.Float32, _atypes.Float64, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.UInt16, _atypes.UInt32, _atypes.UInt64, _atypes.UInt8)
TV_XlaBroadcastHelper_Tindices = TypeVar("TV_XlaBroadcastHelper_Tindices", _atypes.Int32, _atypes.Int64)

@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('xla_broadcast_helper')
def xla_broadcast_helper(lhs: Annotated[Any, TV_XlaBroadcastHelper_T], rhs: Annotated[Any, TV_XlaBroadcastHelper_T], broadcast_dims: Annotated[Any, TV_XlaBroadcastHelper_Tindices], name=None):
  r"""Helper operator for performing XLA-style broadcasts

  Broadcasts `lhs` and `rhs` to the same rank, by adding size 1 dimensions to
  whichever of `lhs` and `rhs` has the lower rank, using XLA's broadcasting rules
  for binary operators.

  Args:
    lhs: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`, `complex64`, `int64`, `qint8`, `quint8`, `qint32`, `bfloat16`, `qint16`, `quint16`, `uint16`, `complex128`, `half`, `uint32`, `uint64`.
      the LHS input tensor
    rhs: A `Tensor`. Must have the same type as `lhs`. the RHS input tensor
    broadcast_dims: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      an XLA-style broadcast dimension specification
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (lhs_output, rhs_output).

    lhs_output: A `Tensor`. Has the same type as `lhs`. the broadcasted LHS tensor
    rhs_output: A `Tensor`. Has the same type as `lhs`. the broadcasted RHS tensor
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "XlaBroadcastHelper", name, lhs, rhs, broadcast_dims)
      _result = _XlaBroadcastHelperOutput._make(_result)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_xla_broadcast_helper(
          (lhs, rhs, broadcast_dims, name,), None)
      if _result is not NotImplemented:
        return _result
      return xla_broadcast_helper_eager_fallback(
          lhs, rhs, broadcast_dims, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            xla_broadcast_helper, (), dict(lhs=lhs, rhs=rhs,
                                           broadcast_dims=broadcast_dims,
                                           name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_xla_broadcast_helper(
        (lhs, rhs, broadcast_dims, name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "XlaBroadcastHelper", lhs=lhs, rhs=rhs, broadcast_dims=broadcast_dims,
                              name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          xla_broadcast_helper, (), dict(lhs=lhs, rhs=rhs,
                                         broadcast_dims=broadcast_dims,
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
        "XlaBroadcastHelper", _inputs_flat, _attrs, _result)
  _result = _XlaBroadcastHelperOutput._make(_result)
  return _result

XlaBroadcastHelper = tf_export("raw_ops.XlaBroadcastHelper")(_ops.to_raw_op(xla_broadcast_helper))
_dispatcher_for_xla_broadcast_helper = xla_broadcast_helper._tf_type_based_dispatcher.Dispatch


def xla_broadcast_helper_eager_fallback(lhs: Annotated[Any, TV_XlaBroadcastHelper_T], rhs: Annotated[Any, TV_XlaBroadcastHelper_T], broadcast_dims: Annotated[Any, TV_XlaBroadcastHelper_Tindices], name, ctx):
  _attr_T, _inputs_T = _execute.args_to_matching_eager([lhs, rhs], ctx, [_dtypes.float32, _dtypes.float64, _dtypes.int32, _dtypes.uint8, _dtypes.int16, _dtypes.int8, _dtypes.complex64, _dtypes.int64, _dtypes.qint8, _dtypes.quint8, _dtypes.qint32, _dtypes.bfloat16, _dtypes.qint16, _dtypes.quint16, _dtypes.uint16, _dtypes.complex128, _dtypes.half, _dtypes.uint32, _dtypes.uint64, ])
  (lhs, rhs) = _inputs_T
  _attr_Tindices, (broadcast_dims,) = _execute.args_to_matching_eager([broadcast_dims], ctx, [_dtypes.int32, _dtypes.int64, ])
  _inputs_flat = [lhs, rhs, broadcast_dims]
  _attrs = ("T", _attr_T, "Tindices", _attr_Tindices)
  _result = _execute.execute(b"XlaBroadcastHelper", 2, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "XlaBroadcastHelper", _inputs_flat, _attrs, _result)
  _result = _XlaBroadcastHelperOutput._make(_result)
  return _result


@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('xla_call_module')
def xla_call_module(args, version: int, module: str, Sout, Tout, dim_args_spec=[], platforms=[], function_list=[], has_token_input_output:bool=False, disabled_checks=[], name=None):
  r"""Invokes a StableHLO module.

  This op is used with JAX native serialization in a TensorFlow context with
  stability guarantees.

  Args:
    args: A list of `Tensor` objects.
      A list of `Tensor` with possibly different types to be passed as arguments
      to the `module`. These are the actual arguments and do not include the
      platform argument (see `platforms`) nor the dimension arguments (see
      `dim_args_spec`).
    version: An `int`.
      Tracks changes the semantics of the op, to support backwards
      compatibility. Minimum supported version is 2. From
      version 2, the op carries a StableHLO text or bytecode `module`. From
      version 3, the op also supports the `platforms` attribute. From version 4,
      the op carries a StableHLO module with compatibility guarantees. From version
      5, XLACallModule can include `stablehlo.custom_call` op to execute tf
      functions. From version 6 the op supports the `disabled_checks` attribute.
      See more versioning details at https://github.com/search?q=repo%3Atensorflow%2Ftensorflow+path%3Axla_call_module+%22int+kVersionMaximumSupported%22&type=code.
    module: A `string`.
      A serialized computation, a text or bytecode representation of
      an mlir.Module. The return type must be a tuple if and only if the `Sout` is
      a list with 0 or more than 1 elements. The length of `Tout` and
      `Sout` must match. This op always returns a tuple of results, even if the
      module returns a single result.
    Sout: A list of shapes (each a `tf.TensorShape` or list of `ints`).
      List of output tensor shapes.
    Tout: A list of `tf.DTypes`. List of output tensor data types.
    dim_args_spec: An optional list of `strings`. Defaults to `[]`.
      this attribute is not supported anymore.
    platforms: An optional list of `strings`. Defaults to `[]`.
      the list of platforms supported by `module`. The list can contain
      the strings "CPU", "CUDA", "ROCM", or "TPU". It is an error to compile
      this op for a platform that does not appear in the list. This check can be
      disabled using `disabled_checks`. If the list contains more than
      one platform, then the `module` takes one additional 0-dimensional
      integer-tensor parameter in the first position, encoding the index in
      `platforms` of the current compilation platform. This parameter has value 0
      if the plaform is not among `platforms` and the check has been disabled.
      The list can be empty in old versions (earlier than 6) to denote that no
      platform checking must be performed at loading time.
    function_list: An optional list of functions decorated with @Defun. Defaults to `[]`.
      This list contains the TensorFlow FunctionDefs that are used by
      the XLACallModule. If the XLACallModule contains `stablehlo.custom_call`
      operations, they can call TensorFlow graph functions outside of the
      XLACallModule. This `function_list` attribute registers the dependency of the
      XLACallModule on those functions. This attribute was added in version 5.
    has_token_input_output: An optional `bool`. Defaults to `False`.
      If true, the embedded StableHLO module's main function
      must take a `!stablehlo.token` as its first argument and returns a token as
      its first result. This can be used in conjunction with the TF2XLA's side
      effect mechanism in order to model side effects. This is used only in versions
      prior to version 9. After that, the number and position of tokens among
      the arguments and results are obtained from the main function type. This
      allows us to support more than one token and not necessarily at the start.
    disabled_checks: An optional list of `strings`. Defaults to `[]`.
      A list of strings describing the safety checks that were
      disabled at serialization time. This attribute was added in version 6.
      For more details see
      https://github.com/search?q=repo%3Agoogle%2Fjax+path%3Ajax_export+%22class+DisabledSafetyCheck%22&type=code.
      This list, supplemented with a comma-separate list of directives specified
      using the flag --tf_xla_call_module_disabled_checks,
      is used at module loading time to skip the corresponding checks.
    name: A name for the operation (optional).

  Returns:
    A list of `Tensor` objects of type `Tout`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "XlaCallModule", name, args, "version", version, "module",
        module, "Sout", Sout, "Tout", Tout, "dim_args_spec", dim_args_spec,
        "platforms", platforms, "function_list", function_list,
        "has_token_input_output", has_token_input_output, "disabled_checks",
        disabled_checks)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_xla_call_module(
          (args, version, module, Sout, Tout, dim_args_spec, platforms,
          function_list, has_token_input_output, disabled_checks, name,), None)
      if _result is not NotImplemented:
        return _result
      return xla_call_module_eager_fallback(
          args, version=version, module=module, Sout=Sout, Tout=Tout,
          dim_args_spec=dim_args_spec, platforms=platforms,
          function_list=function_list,
          has_token_input_output=has_token_input_output,
          disabled_checks=disabled_checks, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            xla_call_module, (), dict(args=args, version=version,
                                      module=module, Sout=Sout, Tout=Tout,
                                      dim_args_spec=dim_args_spec,
                                      platforms=platforms,
                                      function_list=function_list,
                                      has_token_input_output=has_token_input_output,
                                      disabled_checks=disabled_checks,
                                      name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_xla_call_module(
        (args, version, module, Sout, Tout, dim_args_spec, platforms,
        function_list, has_token_input_output, disabled_checks, name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  version = _execute.make_int(version, "version")
  module = _execute.make_str(module, "module")
  if not isinstance(Sout, (list, tuple)):
    raise TypeError(
        "Expected list for 'Sout' argument to "
        "'xla_call_module' Op, not %r." % Sout)
  Sout = [_execute.make_shape(_s, "Sout") for _s in Sout]
  if not isinstance(Tout, (list, tuple)):
    raise TypeError(
        "Expected list for 'Tout' argument to "
        "'xla_call_module' Op, not %r." % Tout)
  Tout = [_execute.make_type(_t, "Tout") for _t in Tout]
  if dim_args_spec is None:
    dim_args_spec = []
  if not isinstance(dim_args_spec, (list, tuple)):
    raise TypeError(
        "Expected list for 'dim_args_spec' argument to "
        "'xla_call_module' Op, not %r." % dim_args_spec)
  dim_args_spec = [_execute.make_str(_s, "dim_args_spec") for _s in dim_args_spec]
  if platforms is None:
    platforms = []
  if not isinstance(platforms, (list, tuple)):
    raise TypeError(
        "Expected list for 'platforms' argument to "
        "'xla_call_module' Op, not %r." % platforms)
  platforms = [_execute.make_str(_s, "platforms") for _s in platforms]
  if function_list is None:
    function_list = []
  if not isinstance(function_list, (list, tuple)):
    raise TypeError(
        "Expected list for 'function_list' argument to "
        "'xla_call_module' Op, not %r." % function_list)
  if has_token_input_output is None:
    has_token_input_output = False
  has_token_input_output = _execute.make_bool(has_token_input_output, "has_token_input_output")
  if disabled_checks is None:
    disabled_checks = []
  if not isinstance(disabled_checks, (list, tuple)):
    raise TypeError(
        "Expected list for 'disabled_checks' argument to "
        "'xla_call_module' Op, not %r." % disabled_checks)
  disabled_checks = [_execute.make_str(_s, "disabled_checks") for _s in disabled_checks]
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "XlaCallModule", args=args, version=version, module=module, Sout=Sout,
                         Tout=Tout, dim_args_spec=dim_args_spec,
                         platforms=platforms, function_list=function_list,
                         has_token_input_output=has_token_input_output,
                         disabled_checks=disabled_checks, name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          xla_call_module, (), dict(args=args, version=version, module=module,
                                    Sout=Sout, Tout=Tout,
                                    dim_args_spec=dim_args_spec,
                                    platforms=platforms,
                                    function_list=function_list,
                                    has_token_input_output=has_token_input_output,
                                    disabled_checks=disabled_checks,
                                    name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  _result = _outputs[:]
  if not _result:
    return _op
  if _execute.must_record_gradient():
    _attrs = ("version", _op._get_attr_int("version"), "module",
              _op.get_attr("module"), "Sout", _op.get_attr("Sout"), "Tout",
              _op.get_attr("Tout"), "Tin", _op.get_attr("Tin"),
              "dim_args_spec", _op.get_attr("dim_args_spec"), "platforms",
              _op.get_attr("platforms"), "function_list",
              _op.get_attr("function_list"), "has_token_input_output",
              _op._get_attr_bool("has_token_input_output"), "disabled_checks",
              _op.get_attr("disabled_checks"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "XlaCallModule", _inputs_flat, _attrs, _result)
  return _result

XlaCallModule = tf_export("raw_ops.XlaCallModule")(_ops.to_raw_op(xla_call_module))
_dispatcher_for_xla_call_module = xla_call_module._tf_type_based_dispatcher.Dispatch


def xla_call_module_eager_fallback(args, version: int, module: str, Sout, Tout, dim_args_spec, platforms, function_list, has_token_input_output: bool, disabled_checks, name, ctx):
  version = _execute.make_int(version, "version")
  module = _execute.make_str(module, "module")
  if not isinstance(Sout, (list, tuple)):
    raise TypeError(
        "Expected list for 'Sout' argument to "
        "'xla_call_module' Op, not %r." % Sout)
  Sout = [_execute.make_shape(_s, "Sout") for _s in Sout]
  if not isinstance(Tout, (list, tuple)):
    raise TypeError(
        "Expected list for 'Tout' argument to "
        "'xla_call_module' Op, not %r." % Tout)
  Tout = [_execute.make_type(_t, "Tout") for _t in Tout]
  if dim_args_spec is None:
    dim_args_spec = []
  if not isinstance(dim_args_spec, (list, tuple)):
    raise TypeError(
        "Expected list for 'dim_args_spec' argument to "
        "'xla_call_module' Op, not %r." % dim_args_spec)
  dim_args_spec = [_execute.make_str(_s, "dim_args_spec") for _s in dim_args_spec]
  if platforms is None:
    platforms = []
  if not isinstance(platforms, (list, tuple)):
    raise TypeError(
        "Expected list for 'platforms' argument to "
        "'xla_call_module' Op, not %r." % platforms)
  platforms = [_execute.make_str(_s, "platforms") for _s in platforms]
  if function_list is None:
    function_list = []
  if not isinstance(function_list, (list, tuple)):
    raise TypeError(
        "Expected list for 'function_list' argument to "
        "'xla_call_module' Op, not %r." % function_list)
  if has_token_input_output is None:
    has_token_input_output = False
  has_token_input_output = _execute.make_bool(has_token_input_output, "has_token_input_output")
  if disabled_checks is None:
    disabled_checks = []
  if not isinstance(disabled_checks, (list, tuple)):
    raise TypeError(
        "Expected list for 'disabled_checks' argument to "
        "'xla_call_module' Op, not %r." % disabled_checks)
  disabled_checks = [_execute.make_str(_s, "disabled_checks") for _s in disabled_checks]
  _attr_Tin, args = _execute.convert_to_mixed_eager_tensors(args, ctx)
  _inputs_flat = list(args)
  _attrs = ("version", version, "module", module, "Sout", Sout, "Tout", Tout,
  "Tin", _attr_Tin, "dim_args_spec", dim_args_spec, "platforms", platforms,
  "function_list", function_list, "has_token_input_output",
  has_token_input_output, "disabled_checks", disabled_checks)
  _result = _execute.execute(b"XlaCallModule", len(Tout), inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "XlaCallModule", _inputs_flat, _attrs, _result)
  return _result


TV_XlaConv_T = TypeVar("TV_XlaConv_T", _atypes.BFloat16, _atypes.Complex128, _atypes.Complex64, _atypes.Float32, _atypes.Float64, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.UInt16, _atypes.UInt32, _atypes.UInt64, _atypes.UInt8)
TV_XlaConv_Tindices = TypeVar("TV_XlaConv_Tindices", _atypes.Int32, _atypes.Int64)

@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('xla_conv')
def xla_conv(lhs: Annotated[Any, TV_XlaConv_T], rhs: Annotated[Any, TV_XlaConv_T], window_strides: Annotated[Any, TV_XlaConv_Tindices], padding: Annotated[Any, TV_XlaConv_Tindices], lhs_dilation: Annotated[Any, TV_XlaConv_Tindices], rhs_dilation: Annotated[Any, TV_XlaConv_Tindices], feature_group_count: Annotated[Any, TV_XlaConv_Tindices], dimension_numbers: str, precision_config: str, name=None) -> Annotated[Any, TV_XlaConv_T]:
  r"""Wraps the XLA ConvGeneralDilated operator, documented at

   https://www.tensorflow.org/performance/xla/operation_semantics#conv_convolution
  .

  Args:
    lhs: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`, `complex64`, `int64`, `qint8`, `quint8`, `qint32`, `bfloat16`, `qint16`, `quint16`, `uint16`, `complex128`, `half`, `uint32`, `uint64`.
      the input tensor
    rhs: A `Tensor`. Must have the same type as `lhs`. the kernel tensor
    window_strides: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      the inter-window strides
    padding: A `Tensor`. Must have the same type as `window_strides`.
      the padding to apply at the start and end of each input dimensions
    lhs_dilation: A `Tensor`. Must have the same type as `window_strides`.
      dilation to apply between input elements
    rhs_dilation: A `Tensor`. Must have the same type as `window_strides`.
      dilation to apply between kernel elements
    feature_group_count: A `Tensor`. Must have the same type as `window_strides`.
      number of feature groups for grouped convolution.
    dimension_numbers: A `string`.
      a serialized xla::ConvolutionDimensionNumbers proto.
    precision_config: A `string`. a serialized xla::PrecisionConfig proto.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `lhs`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "XlaConv", name, lhs, rhs, window_strides, padding,
        lhs_dilation, rhs_dilation, feature_group_count, "dimension_numbers",
        dimension_numbers, "precision_config", precision_config)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_xla_conv(
          (lhs, rhs, window_strides, padding, lhs_dilation, rhs_dilation,
          feature_group_count, dimension_numbers, precision_config, name,),
          None)
      if _result is not NotImplemented:
        return _result
      return xla_conv_eager_fallback(
          lhs, rhs, window_strides, padding, lhs_dilation, rhs_dilation,
          feature_group_count, dimension_numbers=dimension_numbers,
          precision_config=precision_config, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            xla_conv, (), dict(lhs=lhs, rhs=rhs,
                               window_strides=window_strides, padding=padding,
                               lhs_dilation=lhs_dilation,
                               rhs_dilation=rhs_dilation,
                               feature_group_count=feature_group_count,
                               dimension_numbers=dimension_numbers,
                               precision_config=precision_config, name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_xla_conv(
        (lhs, rhs, window_strides, padding, lhs_dilation, rhs_dilation,
        feature_group_count, dimension_numbers, precision_config, name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  dimension_numbers = _execute.make_str(dimension_numbers, "dimension_numbers")
  precision_config = _execute.make_str(precision_config, "precision_config")
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "XlaConv", lhs=lhs, rhs=rhs, window_strides=window_strides,
                   padding=padding, lhs_dilation=lhs_dilation,
                   rhs_dilation=rhs_dilation,
                   feature_group_count=feature_group_count,
                   dimension_numbers=dimension_numbers,
                   precision_config=precision_config, name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          xla_conv, (), dict(lhs=lhs, rhs=rhs, window_strides=window_strides,
                             padding=padding, lhs_dilation=lhs_dilation,
                             rhs_dilation=rhs_dilation,
                             feature_group_count=feature_group_count,
                             dimension_numbers=dimension_numbers,
                             precision_config=precision_config, name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"), "Tindices",
              _op._get_attr_type("Tindices"), "dimension_numbers",
              _op.get_attr("dimension_numbers"), "precision_config",
              _op.get_attr("precision_config"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "XlaConv", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

XlaConv = tf_export("raw_ops.XlaConv")(_ops.to_raw_op(xla_conv))
_dispatcher_for_xla_conv = xla_conv._tf_type_based_dispatcher.Dispatch


def xla_conv_eager_fallback(lhs: Annotated[Any, TV_XlaConv_T], rhs: Annotated[Any, TV_XlaConv_T], window_strides: Annotated[Any, TV_XlaConv_Tindices], padding: Annotated[Any, TV_XlaConv_Tindices], lhs_dilation: Annotated[Any, TV_XlaConv_Tindices], rhs_dilation: Annotated[Any, TV_XlaConv_Tindices], feature_group_count: Annotated[Any, TV_XlaConv_Tindices], dimension_numbers: str, precision_config: str, name, ctx) -> Annotated[Any, TV_XlaConv_T]:
  dimension_numbers = _execute.make_str(dimension_numbers, "dimension_numbers")
  precision_config = _execute.make_str(precision_config, "precision_config")
  _attr_T, _inputs_T = _execute.args_to_matching_eager([lhs, rhs], ctx, [_dtypes.float32, _dtypes.float64, _dtypes.int32, _dtypes.uint8, _dtypes.int16, _dtypes.int8, _dtypes.complex64, _dtypes.int64, _dtypes.qint8, _dtypes.quint8, _dtypes.qint32, _dtypes.bfloat16, _dtypes.qint16, _dtypes.quint16, _dtypes.uint16, _dtypes.complex128, _dtypes.half, _dtypes.uint32, _dtypes.uint64, ])
  (lhs, rhs) = _inputs_T
  _attr_Tindices, _inputs_Tindices = _execute.args_to_matching_eager([window_strides, padding, lhs_dilation, rhs_dilation, feature_group_count], ctx, [_dtypes.int32, _dtypes.int64, ])
  (window_strides, padding, lhs_dilation, rhs_dilation, feature_group_count) = _inputs_Tindices
  _inputs_flat = [lhs, rhs, window_strides, padding, lhs_dilation, rhs_dilation, feature_group_count]
  _attrs = ("T", _attr_T, "Tindices", _attr_Tindices, "dimension_numbers",
  dimension_numbers, "precision_config", precision_config)
  _result = _execute.execute(b"XlaConv", 1, inputs=_inputs_flat, attrs=_attrs,
                             ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "XlaConv", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_XlaConvV2_LhsT = TypeVar("TV_XlaConvV2_LhsT", _atypes.BFloat16, _atypes.Complex128, _atypes.Complex64, _atypes.Float32, _atypes.Float64, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.UInt16, _atypes.UInt32, _atypes.UInt64, _atypes.UInt8)
TV_XlaConvV2_RhsT = TypeVar("TV_XlaConvV2_RhsT", _atypes.BFloat16, _atypes.Complex128, _atypes.Complex64, _atypes.Float32, _atypes.Float64, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.UInt16, _atypes.UInt32, _atypes.UInt64, _atypes.UInt8)
TV_XlaConvV2_Tindices = TypeVar("TV_XlaConvV2_Tindices", _atypes.Int32, _atypes.Int64)
TV_XlaConvV2_preferred_element_type = TypeVar("TV_XlaConvV2_preferred_element_type", _atypes.BFloat16, _atypes.Complex128, _atypes.Complex64, _atypes.Float32, _atypes.Float64, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.UInt16, _atypes.UInt32, _atypes.UInt64, _atypes.UInt8)

@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('xla_conv_v2')
def xla_conv_v2(lhs: Annotated[Any, TV_XlaConvV2_LhsT], rhs: Annotated[Any, TV_XlaConvV2_RhsT], window_strides: Annotated[Any, TV_XlaConvV2_Tindices], padding: Annotated[Any, TV_XlaConvV2_Tindices], lhs_dilation: Annotated[Any, TV_XlaConvV2_Tindices], rhs_dilation: Annotated[Any, TV_XlaConvV2_Tindices], feature_group_count: Annotated[Any, TV_XlaConvV2_Tindices], dimension_numbers: str, precision_config: str, preferred_element_type: TV_XlaConvV2_preferred_element_type, batch_group_count:int=1, name=None) -> Annotated[Any, TV_XlaConvV2_preferred_element_type]:
  r"""Wraps the XLA ConvGeneralDilated operator, documented at

   https://www.tensorflow.org/performance/xla/operation_semantics#conv_convolution
  .

  Args:
    lhs: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`, `complex64`, `int64`, `qint8`, `quint8`, `qint32`, `bfloat16`, `qint16`, `quint16`, `uint16`, `complex128`, `half`, `uint32`, `uint64`.
      input tensor
    rhs: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`, `complex64`, `int64`, `qint8`, `quint8`, `qint32`, `bfloat16`, `qint16`, `quint16`, `uint16`, `complex128`, `half`, `uint32`, `uint64`.
      kernel tensor
    window_strides: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      inter-window strides
    padding: A `Tensor`. Must have the same type as `window_strides`.
      padding to apply at the start and end of each input dimensions
    lhs_dilation: A `Tensor`. Must have the same type as `window_strides`.
      dilation to apply between input elements
    rhs_dilation: A `Tensor`. Must have the same type as `window_strides`.
      dilation to apply between kernel elements
    feature_group_count: A `Tensor`. Must have the same type as `window_strides`.
      number of feature groups for grouped convolution.
    dimension_numbers: A `string`.
      serialized xla::ConvolutionDimensionNumbers proto.
    precision_config: A `string`. serialized xla::PrecisionConfig proto.
    preferred_element_type: A `tf.DType` from: `tf.float32, tf.float64, tf.int32, tf.uint8, tf.int16, tf.int8, tf.complex64, tf.int64, tf.qint8, tf.quint8, tf.qint32, tf.bfloat16, tf.qint16, tf.quint16, tf.uint16, tf.complex128, tf.half, tf.uint32, tf.uint64`.
      type of the tensor.
    batch_group_count: An optional `int`. Defaults to `1`.
      number of batch groups or grouped filters.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `preferred_element_type`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "XlaConvV2", name, lhs, rhs, window_strides, padding,
        lhs_dilation, rhs_dilation, feature_group_count, "dimension_numbers",
        dimension_numbers, "precision_config", precision_config,
        "preferred_element_type", preferred_element_type, "batch_group_count",
        batch_group_count)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_xla_conv_v2(
          (lhs, rhs, window_strides, padding, lhs_dilation, rhs_dilation,
          feature_group_count, dimension_numbers, precision_config,
          preferred_element_type, batch_group_count, name,), None)
      if _result is not NotImplemented:
        return _result
      return xla_conv_v2_eager_fallback(
          lhs, rhs, window_strides, padding, lhs_dilation, rhs_dilation,
          feature_group_count, dimension_numbers=dimension_numbers,
          precision_config=precision_config,
          preferred_element_type=preferred_element_type,
          batch_group_count=batch_group_count, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            xla_conv_v2, (), dict(lhs=lhs, rhs=rhs,
                                  window_strides=window_strides,
                                  padding=padding, lhs_dilation=lhs_dilation,
                                  rhs_dilation=rhs_dilation,
                                  feature_group_count=feature_group_count,
                                  dimension_numbers=dimension_numbers,
                                  precision_config=precision_config,
                                  preferred_element_type=preferred_element_type,
                                  batch_group_count=batch_group_count,
                                  name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_xla_conv_v2(
        (lhs, rhs, window_strides, padding, lhs_dilation, rhs_dilation,
        feature_group_count, dimension_numbers, precision_config,
        preferred_element_type, batch_group_count, name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  dimension_numbers = _execute.make_str(dimension_numbers, "dimension_numbers")
  precision_config = _execute.make_str(precision_config, "precision_config")
  preferred_element_type = _execute.make_type(preferred_element_type, "preferred_element_type")
  if batch_group_count is None:
    batch_group_count = 1
  batch_group_count = _execute.make_int(batch_group_count, "batch_group_count")
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "XlaConvV2", lhs=lhs, rhs=rhs, window_strides=window_strides,
                     padding=padding, lhs_dilation=lhs_dilation,
                     rhs_dilation=rhs_dilation,
                     feature_group_count=feature_group_count,
                     dimension_numbers=dimension_numbers,
                     precision_config=precision_config,
                     preferred_element_type=preferred_element_type,
                     batch_group_count=batch_group_count, name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          xla_conv_v2, (), dict(lhs=lhs, rhs=rhs,
                                window_strides=window_strides,
                                padding=padding, lhs_dilation=lhs_dilation,
                                rhs_dilation=rhs_dilation,
                                feature_group_count=feature_group_count,
                                dimension_numbers=dimension_numbers,
                                precision_config=precision_config,
                                preferred_element_type=preferred_element_type,
                                batch_group_count=batch_group_count,
                                name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("LhsT", _op._get_attr_type("LhsT"), "RhsT",
              _op._get_attr_type("RhsT"), "Tindices",
              _op._get_attr_type("Tindices"), "dimension_numbers",
              _op.get_attr("dimension_numbers"), "precision_config",
              _op.get_attr("precision_config"), "preferred_element_type",
              _op._get_attr_type("preferred_element_type"),
              "batch_group_count", _op._get_attr_int("batch_group_count"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "XlaConvV2", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

XlaConvV2 = tf_export("raw_ops.XlaConvV2")(_ops.to_raw_op(xla_conv_v2))
_dispatcher_for_xla_conv_v2 = xla_conv_v2._tf_type_based_dispatcher.Dispatch


def xla_conv_v2_eager_fallback(lhs: Annotated[Any, TV_XlaConvV2_LhsT], rhs: Annotated[Any, TV_XlaConvV2_RhsT], window_strides: Annotated[Any, TV_XlaConvV2_Tindices], padding: Annotated[Any, TV_XlaConvV2_Tindices], lhs_dilation: Annotated[Any, TV_XlaConvV2_Tindices], rhs_dilation: Annotated[Any, TV_XlaConvV2_Tindices], feature_group_count: Annotated[Any, TV_XlaConvV2_Tindices], dimension_numbers: str, precision_config: str, preferred_element_type: TV_XlaConvV2_preferred_element_type, batch_group_count: int, name, ctx) -> Annotated[Any, TV_XlaConvV2_preferred_element_type]:
  dimension_numbers = _execute.make_str(dimension_numbers, "dimension_numbers")
  precision_config = _execute.make_str(precision_config, "precision_config")
  preferred_element_type = _execute.make_type(preferred_element_type, "preferred_element_type")
  if batch_group_count is None:
    batch_group_count = 1
  batch_group_count = _execute.make_int(batch_group_count, "batch_group_count")
  _attr_LhsT, (lhs,) = _execute.args_to_matching_eager([lhs], ctx, [_dtypes.float32, _dtypes.float64, _dtypes.int32, _dtypes.uint8, _dtypes.int16, _dtypes.int8, _dtypes.complex64, _dtypes.int64, _dtypes.qint8, _dtypes.quint8, _dtypes.qint32, _dtypes.bfloat16, _dtypes.qint16, _dtypes.quint16, _dtypes.uint16, _dtypes.complex128, _dtypes.half, _dtypes.uint32, _dtypes.uint64, ])
  _attr_RhsT, (rhs,) = _execute.args_to_matching_eager([rhs], ctx, [_dtypes.float32, _dtypes.float64, _dtypes.int32, _dtypes.uint8, _dtypes.int16, _dtypes.int8, _dtypes.complex64, _dtypes.int64, _dtypes.qint8, _dtypes.quint8, _dtypes.qint32, _dtypes.bfloat16, _dtypes.qint16, _dtypes.quint16, _dtypes.uint16, _dtypes.complex128, _dtypes.half, _dtypes.uint32, _dtypes.uint64, ])
  _attr_Tindices, _inputs_Tindices = _execute.args_to_matching_eager([window_strides, padding, lhs_dilation, rhs_dilation, feature_group_count], ctx, [_dtypes.int32, _dtypes.int64, ])
  (window_strides, padding, lhs_dilation, rhs_dilation, feature_group_count) = _inputs_Tindices
  _inputs_flat = [lhs, rhs, window_strides, padding, lhs_dilation, rhs_dilation, feature_group_count]
  _attrs = ("LhsT", _attr_LhsT, "RhsT", _attr_RhsT, "Tindices",
  _attr_Tindices, "dimension_numbers", dimension_numbers, "precision_config",
  precision_config, "preferred_element_type", preferred_element_type,
  "batch_group_count", batch_group_count)
  _result = _execute.execute(b"XlaConvV2", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "XlaConvV2", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_XlaCustomCall_dtype = TypeVar("TV_XlaCustomCall_dtype", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float16, _atypes.Float32, _atypes.Float64, _atypes.Float8e4m3fn, _atypes.Float8e5m2, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int4, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.Resource, _atypes.String, _atypes.UInt16, _atypes.UInt32, _atypes.UInt4, _atypes.UInt64, _atypes.UInt8, _atypes.Variant)

@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('xla_custom_call')
def xla_custom_call(args, target_name: str, backend_config: str, dtype: TV_XlaCustomCall_dtype, shape, name=None) -> Annotated[Any, TV_XlaCustomCall_dtype]:
  r"""Wraps the XLA CustomCall operator

    documented at https://www.tensorflow.org/xla/operation_semantics#customcall.

  Args:
    args: A list of `Tensor` objects.
      A list of `Tensor` with possibly different types.
    target_name: A `string`.
      Name of the function. A call instruction will be emitted which
      targets this symbol name.
    backend_config: A `string`.
      String, used to encode serialized metadata to the backend.
    dtype: A `tf.DType`. Output tensor data type.
    shape: A `tf.TensorShape` or list of `ints`. Output tensor shape.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `dtype`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "XlaCustomCall", name, args, "target_name", target_name,
        "backend_config", backend_config, "dtype", dtype, "shape", shape)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_xla_custom_call(
          (args, target_name, backend_config, dtype, shape, name,), None)
      if _result is not NotImplemented:
        return _result
      return xla_custom_call_eager_fallback(
          args, target_name=target_name, backend_config=backend_config,
          dtype=dtype, shape=shape, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            xla_custom_call, (), dict(args=args, target_name=target_name,
                                      backend_config=backend_config,
                                      dtype=dtype, shape=shape, name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_xla_custom_call(
        (args, target_name, backend_config, dtype, shape, name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  target_name = _execute.make_str(target_name, "target_name")
  backend_config = _execute.make_str(backend_config, "backend_config")
  dtype = _execute.make_type(dtype, "dtype")
  shape = _execute.make_shape(shape, "shape")
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "XlaCustomCall", args=args, target_name=target_name,
                         backend_config=backend_config, dtype=dtype,
                         shape=shape, name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          xla_custom_call, (), dict(args=args, target_name=target_name,
                                    backend_config=backend_config,
                                    dtype=dtype, shape=shape, name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("target_name", _op.get_attr("target_name"), "backend_config",
              _op.get_attr("backend_config"), "T", _op.get_attr("T"), "dtype",
              _op._get_attr_type("dtype"), "shape", _op.get_attr("shape"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "XlaCustomCall", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

XlaCustomCall = tf_export("raw_ops.XlaCustomCall")(_ops.to_raw_op(xla_custom_call))
_dispatcher_for_xla_custom_call = xla_custom_call._tf_type_based_dispatcher.Dispatch


def xla_custom_call_eager_fallback(args, target_name: str, backend_config: str, dtype: TV_XlaCustomCall_dtype, shape, name, ctx) -> Annotated[Any, TV_XlaCustomCall_dtype]:
  target_name = _execute.make_str(target_name, "target_name")
  backend_config = _execute.make_str(backend_config, "backend_config")
  dtype = _execute.make_type(dtype, "dtype")
  shape = _execute.make_shape(shape, "shape")
  _attr_T, args = _execute.convert_to_mixed_eager_tensors(args, ctx)
  _inputs_flat = list(args)
  _attrs = ("target_name", target_name, "backend_config", backend_config, "T",
  _attr_T, "dtype", dtype, "shape", shape)
  _result = _execute.execute(b"XlaCustomCall", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "XlaCustomCall", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('xla_custom_call_v2')
def xla_custom_call_v2(operands, call_target_name: str, backend_config: str, has_side_effect: bool, result_dtypes, result_shapes, name=None):
  r"""Emits an HLO `CustomCall` operation with multiple outputs.

  As opposed to `XlaCustomCall`, this operation supports multiple outputs.

  See `CustomCall` specification at
    https://tensorflow.org/xla/operation_semantics#customcall,
  and `mhlo.custom_call` specification at
    https://tensorflow.org/mlir/hlo_ops#mhlocustom_call_mlirmhlocustomcallop.

  Args:
    operands: A list of `Tensor` objects.
      A sequence of tensors with possibly different types.
    call_target_name: A `string`.
      Name of the user function. The function signature must conform
      to version 3 of the API, see `API_VERSION_STATUS_RETURNING_UNIFIED`. All
      operands and results assumed to be in the default layout.
    backend_config: A `string`.
      A string that encodes a metadata for the backend.
    has_side_effect: A `bool`.
      Indicates whether the custom call has side effects.
    result_dtypes: A list of `tf.DTypes`. Types of all results.
    result_shapes: A list of shapes (each a `tf.TensorShape` or list of `ints`).
      Shapes of all results.
    name: A name for the operation (optional).

  Returns:
    A list of `Tensor` objects of type `result_dtypes`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "XlaCustomCallV2", name, operands, "call_target_name",
        call_target_name, "backend_config", backend_config, "has_side_effect",
        has_side_effect, "result_dtypes", result_dtypes, "result_shapes",
        result_shapes)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_xla_custom_call_v2(
          (operands, call_target_name, backend_config, has_side_effect,
          result_dtypes, result_shapes, name,), None)
      if _result is not NotImplemented:
        return _result
      return xla_custom_call_v2_eager_fallback(
          operands, call_target_name=call_target_name,
          backend_config=backend_config, has_side_effect=has_side_effect,
          result_dtypes=result_dtypes, result_shapes=result_shapes, name=name,
          ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            xla_custom_call_v2, (), dict(operands=operands,
                                         call_target_name=call_target_name,
                                         backend_config=backend_config,
                                         has_side_effect=has_side_effect,
                                         result_dtypes=result_dtypes,
                                         result_shapes=result_shapes,
                                         name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_xla_custom_call_v2(
        (operands, call_target_name, backend_config, has_side_effect,
        result_dtypes, result_shapes, name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  call_target_name = _execute.make_str(call_target_name, "call_target_name")
  backend_config = _execute.make_str(backend_config, "backend_config")
  has_side_effect = _execute.make_bool(has_side_effect, "has_side_effect")
  if not isinstance(result_dtypes, (list, tuple)):
    raise TypeError(
        "Expected list for 'result_dtypes' argument to "
        "'xla_custom_call_v2' Op, not %r." % result_dtypes)
  result_dtypes = [_execute.make_type(_t, "result_dtypes") for _t in result_dtypes]
  if not isinstance(result_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'result_shapes' argument to "
        "'xla_custom_call_v2' Op, not %r." % result_shapes)
  result_shapes = [_execute.make_shape(_s, "result_shapes") for _s in result_shapes]
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "XlaCustomCallV2", operands=operands,
                           call_target_name=call_target_name,
                           backend_config=backend_config,
                           has_side_effect=has_side_effect,
                           result_dtypes=result_dtypes,
                           result_shapes=result_shapes, name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          xla_custom_call_v2, (), dict(operands=operands,
                                       call_target_name=call_target_name,
                                       backend_config=backend_config,
                                       has_side_effect=has_side_effect,
                                       result_dtypes=result_dtypes,
                                       result_shapes=result_shapes, name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("call_target_name", _op.get_attr("call_target_name"),
              "backend_config", _op.get_attr("backend_config"),
              "has_side_effect", _op._get_attr_bool("has_side_effect"),
              "operand_dtypes", _op.get_attr("operand_dtypes"),
              "result_dtypes", _op.get_attr("result_dtypes"), "result_shapes",
              _op.get_attr("result_shapes"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "XlaCustomCallV2", _inputs_flat, _attrs, _result)
  return _result

XlaCustomCallV2 = tf_export("raw_ops.XlaCustomCallV2")(_ops.to_raw_op(xla_custom_call_v2))
_dispatcher_for_xla_custom_call_v2 = xla_custom_call_v2._tf_type_based_dispatcher.Dispatch


def xla_custom_call_v2_eager_fallback(operands, call_target_name: str, backend_config: str, has_side_effect: bool, result_dtypes, result_shapes, name, ctx):
  call_target_name = _execute.make_str(call_target_name, "call_target_name")
  backend_config = _execute.make_str(backend_config, "backend_config")
  has_side_effect = _execute.make_bool(has_side_effect, "has_side_effect")
  if not isinstance(result_dtypes, (list, tuple)):
    raise TypeError(
        "Expected list for 'result_dtypes' argument to "
        "'xla_custom_call_v2' Op, not %r." % result_dtypes)
  result_dtypes = [_execute.make_type(_t, "result_dtypes") for _t in result_dtypes]
  if not isinstance(result_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'result_shapes' argument to "
        "'xla_custom_call_v2' Op, not %r." % result_shapes)
  result_shapes = [_execute.make_shape(_s, "result_shapes") for _s in result_shapes]
  _attr_operand_dtypes, operands = _execute.convert_to_mixed_eager_tensors(operands, ctx)
  _inputs_flat = list(operands)
  _attrs = ("call_target_name", call_target_name, "backend_config",
  backend_config, "has_side_effect", has_side_effect, "operand_dtypes",
  _attr_operand_dtypes, "result_dtypes", result_dtypes, "result_shapes",
  result_shapes)
  _result = _execute.execute(b"XlaCustomCallV2", len(result_dtypes),
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "XlaCustomCallV2", _inputs_flat, _attrs, _result)
  return _result


@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('xla_dequantize')
def xla_dequantize(input: Annotated[Any, _atypes.UInt32], min_range: float, max_range: float, mode: str, transpose_output: bool, name=None) -> Annotated[Any, _atypes.BFloat16]:
  r"""Takes the packed uint32 input and unpacks the input to uint8 to do

  Dequantization on device.

  Args:
    input: A `Tensor` of type `uint32`.
      Input tensors whose types is uint32, shape is [d0, ..., dn].
    min_range: A `float`.
      The minimum scalar value possibly produced for the input.
    max_range: A `float`.
      The maximum scalar value possibly produced for the input.
    mode: A `string`.
      String to determine the dequantize mode in {"MIN_COMBINED", "MIN_FIRST", "SCALED"}.
    transpose_output: A `bool`.
      Boolean to determine if output is transposed. transpose_output
      is faster when input is large and rank of input is higher than 1.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `bfloat16`.
    Output tensors whose types is bfloat16. If transpose_output is true,
    output shape is [dn * 4, dn-1, ..., d1, d0]. If transpose_output
    is false, output shape is [d0,..., dn * 4].
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "XlaDequantize", name, input, "min_range", min_range,
        "max_range", max_range, "mode", mode, "transpose_output",
        transpose_output)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_xla_dequantize(
          (input, min_range, max_range, mode, transpose_output, name,), None)
      if _result is not NotImplemented:
        return _result
      return xla_dequantize_eager_fallback(
          input, min_range=min_range, max_range=max_range, mode=mode,
          transpose_output=transpose_output, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            xla_dequantize, (), dict(input=input, min_range=min_range,
                                     max_range=max_range, mode=mode,
                                     transpose_output=transpose_output,
                                     name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_xla_dequantize(
        (input, min_range, max_range, mode, transpose_output, name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  min_range = _execute.make_float(min_range, "min_range")
  max_range = _execute.make_float(max_range, "max_range")
  mode = _execute.make_str(mode, "mode")
  transpose_output = _execute.make_bool(transpose_output, "transpose_output")
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "XlaDequantize", input=input, min_range=min_range,
                         max_range=max_range, mode=mode,
                         transpose_output=transpose_output, name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          xla_dequantize, (), dict(input=input, min_range=min_range,
                                   max_range=max_range, mode=mode,
                                   transpose_output=transpose_output,
                                   name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("min_range", _op.get_attr("min_range"), "max_range",
              _op.get_attr("max_range"), "mode", _op.get_attr("mode"),
              "transpose_output", _op._get_attr_bool("transpose_output"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "XlaDequantize", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

XlaDequantize = tf_export("raw_ops.XlaDequantize")(_ops.to_raw_op(xla_dequantize))
_dispatcher_for_xla_dequantize = xla_dequantize._tf_type_based_dispatcher.Dispatch


def xla_dequantize_eager_fallback(input: Annotated[Any, _atypes.UInt32], min_range: float, max_range: float, mode: str, transpose_output: bool, name, ctx) -> Annotated[Any, _atypes.BFloat16]:
  min_range = _execute.make_float(min_range, "min_range")
  max_range = _execute.make_float(max_range, "max_range")
  mode = _execute.make_str(mode, "mode")
  transpose_output = _execute.make_bool(transpose_output, "transpose_output")
  input = _ops.convert_to_tensor(input, _dtypes.uint32)
  _inputs_flat = [input]
  _attrs = ("min_range", min_range, "max_range", max_range, "mode", mode,
  "transpose_output", transpose_output)
  _result = _execute.execute(b"XlaDequantize", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "XlaDequantize", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_XlaDot_T = TypeVar("TV_XlaDot_T", _atypes.BFloat16, _atypes.Complex128, _atypes.Complex64, _atypes.Float32, _atypes.Float64, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.UInt16, _atypes.UInt32, _atypes.UInt64, _atypes.UInt8)

@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('xla_dot')
def xla_dot(lhs: Annotated[Any, TV_XlaDot_T], rhs: Annotated[Any, TV_XlaDot_T], dimension_numbers: str, precision_config: str, name=None) -> Annotated[Any, TV_XlaDot_T]:
  r"""Wraps the XLA DotGeneral operator, documented at

   https://www.tensorflow.org/performance/xla/operation_semantics#dotgeneral
  .

  Args:
    lhs: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`, `complex64`, `int64`, `qint8`, `quint8`, `qint32`, `bfloat16`, `qint16`, `quint16`, `uint16`, `complex128`, `half`, `uint32`, `uint64`.
      the LHS tensor
    rhs: A `Tensor`. Must have the same type as `lhs`. the RHS tensor
    dimension_numbers: A `string`.
      a serialized xla::DotDimensionNumbers proto.
    precision_config: A `string`. a serialized xla::PrecisionConfig proto.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `lhs`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "XlaDot", name, lhs, rhs, "dimension_numbers",
        dimension_numbers, "precision_config", precision_config)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_xla_dot(
          (lhs, rhs, dimension_numbers, precision_config, name,), None)
      if _result is not NotImplemented:
        return _result
      return xla_dot_eager_fallback(
          lhs, rhs, dimension_numbers=dimension_numbers,
          precision_config=precision_config, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            xla_dot, (), dict(lhs=lhs, rhs=rhs,
                              dimension_numbers=dimension_numbers,
                              precision_config=precision_config, name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_xla_dot(
        (lhs, rhs, dimension_numbers, precision_config, name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  dimension_numbers = _execute.make_str(dimension_numbers, "dimension_numbers")
  precision_config = _execute.make_str(precision_config, "precision_config")
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "XlaDot", lhs=lhs, rhs=rhs, dimension_numbers=dimension_numbers,
                  precision_config=precision_config, name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          xla_dot, (), dict(lhs=lhs, rhs=rhs,
                            dimension_numbers=dimension_numbers,
                            precision_config=precision_config, name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"), "dimension_numbers",
              _op.get_attr("dimension_numbers"), "precision_config",
              _op.get_attr("precision_config"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "XlaDot", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

XlaDot = tf_export("raw_ops.XlaDot")(_ops.to_raw_op(xla_dot))
_dispatcher_for_xla_dot = xla_dot._tf_type_based_dispatcher.Dispatch


def xla_dot_eager_fallback(lhs: Annotated[Any, TV_XlaDot_T], rhs: Annotated[Any, TV_XlaDot_T], dimension_numbers: str, precision_config: str, name, ctx) -> Annotated[Any, TV_XlaDot_T]:
  dimension_numbers = _execute.make_str(dimension_numbers, "dimension_numbers")
  precision_config = _execute.make_str(precision_config, "precision_config")
  _attr_T, _inputs_T = _execute.args_to_matching_eager([lhs, rhs], ctx, [_dtypes.float32, _dtypes.float64, _dtypes.int32, _dtypes.uint8, _dtypes.int16, _dtypes.int8, _dtypes.complex64, _dtypes.int64, _dtypes.qint8, _dtypes.quint8, _dtypes.qint32, _dtypes.bfloat16, _dtypes.qint16, _dtypes.quint16, _dtypes.uint16, _dtypes.complex128, _dtypes.half, _dtypes.uint32, _dtypes.uint64, ])
  (lhs, rhs) = _inputs_T
  _inputs_flat = [lhs, rhs]
  _attrs = ("T", _attr_T, "dimension_numbers", dimension_numbers,
  "precision_config", precision_config)
  _result = _execute.execute(b"XlaDot", 1, inputs=_inputs_flat, attrs=_attrs,
                             ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "XlaDot", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_XlaDotV2_LhsT = TypeVar("TV_XlaDotV2_LhsT", _atypes.BFloat16, _atypes.Complex128, _atypes.Complex64, _atypes.Float32, _atypes.Float64, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.UInt16, _atypes.UInt32, _atypes.UInt64, _atypes.UInt8)
TV_XlaDotV2_RhsT = TypeVar("TV_XlaDotV2_RhsT", _atypes.BFloat16, _atypes.Complex128, _atypes.Complex64, _atypes.Float32, _atypes.Float64, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.UInt16, _atypes.UInt32, _atypes.UInt64, _atypes.UInt8)
TV_XlaDotV2_preferred_element_type = TypeVar("TV_XlaDotV2_preferred_element_type", _atypes.BFloat16, _atypes.Complex128, _atypes.Complex64, _atypes.Float32, _atypes.Float64, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.UInt16, _atypes.UInt32, _atypes.UInt64, _atypes.UInt8)

@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('xla_dot_v2')
def xla_dot_v2(lhs: Annotated[Any, TV_XlaDotV2_LhsT], rhs: Annotated[Any, TV_XlaDotV2_RhsT], dimension_numbers: str, precision_config: str, preferred_element_type: TV_XlaDotV2_preferred_element_type, name=None) -> Annotated[Any, TV_XlaDotV2_preferred_element_type]:
  r"""Wraps the XLA DotGeneral operator, documented at

   https://www.tensorflow.org/performance/xla/operation_semantics#dotgeneral
  .

  Args:
    lhs: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`, `complex64`, `int64`, `qint8`, `quint8`, `qint32`, `bfloat16`, `qint16`, `quint16`, `uint16`, `complex128`, `half`, `uint32`, `uint64`.
      the LHS tensor
    rhs: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`, `complex64`, `int64`, `qint8`, `quint8`, `qint32`, `bfloat16`, `qint16`, `quint16`, `uint16`, `complex128`, `half`, `uint32`, `uint64`.
      the RHS tensor
    dimension_numbers: A `string`.
      a serialized xla::DotDimensionNumbers proto.
    precision_config: A `string`. a serialized xla::PrecisionConfig proto.
    preferred_element_type: A `tf.DType` from: `tf.float32, tf.float64, tf.int32, tf.uint8, tf.int16, tf.int8, tf.complex64, tf.int64, tf.qint8, tf.quint8, tf.qint32, tf.bfloat16, tf.qint16, tf.quint16, tf.uint16, tf.complex128, tf.half, tf.uint32, tf.uint64`.
      The type of the tensor.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `preferred_element_type`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "XlaDotV2", name, lhs, rhs, "dimension_numbers",
        dimension_numbers, "precision_config", precision_config,
        "preferred_element_type", preferred_element_type)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_xla_dot_v2(
          (lhs, rhs, dimension_numbers, precision_config,
          preferred_element_type, name,), None)
      if _result is not NotImplemented:
        return _result
      return xla_dot_v2_eager_fallback(
          lhs, rhs, dimension_numbers=dimension_numbers,
          precision_config=precision_config,
          preferred_element_type=preferred_element_type, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            xla_dot_v2, (), dict(lhs=lhs, rhs=rhs,
                                 dimension_numbers=dimension_numbers,
                                 precision_config=precision_config,
                                 preferred_element_type=preferred_element_type,
                                 name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_xla_dot_v2(
        (lhs, rhs, dimension_numbers, precision_config,
        preferred_element_type, name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  dimension_numbers = _execute.make_str(dimension_numbers, "dimension_numbers")
  precision_config = _execute.make_str(precision_config, "precision_config")
  preferred_element_type = _execute.make_type(preferred_element_type, "preferred_element_type")
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "XlaDotV2", lhs=lhs, rhs=rhs, dimension_numbers=dimension_numbers,
                    precision_config=precision_config,
                    preferred_element_type=preferred_element_type, name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          xla_dot_v2, (), dict(lhs=lhs, rhs=rhs,
                               dimension_numbers=dimension_numbers,
                               precision_config=precision_config,
                               preferred_element_type=preferred_element_type,
                               name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("LhsT", _op._get_attr_type("LhsT"), "RhsT",
              _op._get_attr_type("RhsT"), "dimension_numbers",
              _op.get_attr("dimension_numbers"), "precision_config",
              _op.get_attr("precision_config"), "preferred_element_type",
              _op._get_attr_type("preferred_element_type"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "XlaDotV2", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

XlaDotV2 = tf_export("raw_ops.XlaDotV2")(_ops.to_raw_op(xla_dot_v2))
_dispatcher_for_xla_dot_v2 = xla_dot_v2._tf_type_based_dispatcher.Dispatch


def xla_dot_v2_eager_fallback(lhs: Annotated[Any, TV_XlaDotV2_LhsT], rhs: Annotated[Any, TV_XlaDotV2_RhsT], dimension_numbers: str, precision_config: str, preferred_element_type: TV_XlaDotV2_preferred_element_type, name, ctx) -> Annotated[Any, TV_XlaDotV2_preferred_element_type]:
  dimension_numbers = _execute.make_str(dimension_numbers, "dimension_numbers")
  precision_config = _execute.make_str(precision_config, "precision_config")
  preferred_element_type = _execute.make_type(preferred_element_type, "preferred_element_type")
  _attr_LhsT, (lhs,) = _execute.args_to_matching_eager([lhs], ctx, [_dtypes.float32, _dtypes.float64, _dtypes.int32, _dtypes.uint8, _dtypes.int16, _dtypes.int8, _dtypes.complex64, _dtypes.int64, _dtypes.qint8, _dtypes.quint8, _dtypes.qint32, _dtypes.bfloat16, _dtypes.qint16, _dtypes.quint16, _dtypes.uint16, _dtypes.complex128, _dtypes.half, _dtypes.uint32, _dtypes.uint64, ])
  _attr_RhsT, (rhs,) = _execute.args_to_matching_eager([rhs], ctx, [_dtypes.float32, _dtypes.float64, _dtypes.int32, _dtypes.uint8, _dtypes.int16, _dtypes.int8, _dtypes.complex64, _dtypes.int64, _dtypes.qint8, _dtypes.quint8, _dtypes.qint32, _dtypes.bfloat16, _dtypes.qint16, _dtypes.quint16, _dtypes.uint16, _dtypes.complex128, _dtypes.half, _dtypes.uint32, _dtypes.uint64, ])
  _inputs_flat = [lhs, rhs]
  _attrs = ("LhsT", _attr_LhsT, "RhsT", _attr_RhsT, "dimension_numbers",
  dimension_numbers, "precision_config", precision_config,
  "preferred_element_type", preferred_element_type)
  _result = _execute.execute(b"XlaDotV2", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "XlaDotV2", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_XlaDynamicSlice_T = TypeVar("TV_XlaDynamicSlice_T", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float16, _atypes.Float32, _atypes.Float64, _atypes.Float8e4m3fn, _atypes.Float8e5m2, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int4, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.Resource, _atypes.String, _atypes.UInt16, _atypes.UInt32, _atypes.UInt4, _atypes.UInt64, _atypes.UInt8, _atypes.Variant)
TV_XlaDynamicSlice_Tindices = TypeVar("TV_XlaDynamicSlice_Tindices", _atypes.Int32, _atypes.Int64)

@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('xla_dynamic_slice')
def xla_dynamic_slice(input: Annotated[Any, TV_XlaDynamicSlice_T], start_indices: Annotated[Any, TV_XlaDynamicSlice_Tindices], size_indices: Annotated[Any, TV_XlaDynamicSlice_Tindices], name=None) -> Annotated[Any, TV_XlaDynamicSlice_T]:
  r"""Wraps the XLA DynamicSlice operator, documented at

   https://www.tensorflow.org/performance/xla/operation_semantics#dynamicslice
  .

  DynamicSlice extracts a sub-array from the input array at dynamic
  start_indices. The size of the slice in each dimension is passed in
  size_indices, which specify the end point of exclusive slice intervals in each
  dimension -- [start, start + size). The shape of start_indices must have rank 1,
  with dimension size equal to the rank of operand.

  Args:
    input: A `Tensor`. A `Tensor` of type T.
    start_indices: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      List of N integers containing the slice size for each
      dimension. Each value must be strictly greater than zero, and start + size
      must be less than or equal to the size of the dimension to avoid
      implementation defined behavior.
    size_indices: A `Tensor`. Must have the same type as `start_indices`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "XlaDynamicSlice", name, input, start_indices, size_indices)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_xla_dynamic_slice(
          (input, start_indices, size_indices, name,), None)
      if _result is not NotImplemented:
        return _result
      return xla_dynamic_slice_eager_fallback(
          input, start_indices, size_indices, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            xla_dynamic_slice, (), dict(input=input,
                                        start_indices=start_indices,
                                        size_indices=size_indices, name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_xla_dynamic_slice(
        (input, start_indices, size_indices, name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "XlaDynamicSlice", input=input, start_indices=start_indices,
                           size_indices=size_indices, name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          xla_dynamic_slice, (), dict(input=input,
                                      start_indices=start_indices,
                                      size_indices=size_indices, name=name)
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
        "XlaDynamicSlice", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

XlaDynamicSlice = tf_export("raw_ops.XlaDynamicSlice")(_ops.to_raw_op(xla_dynamic_slice))
_dispatcher_for_xla_dynamic_slice = xla_dynamic_slice._tf_type_based_dispatcher.Dispatch


def xla_dynamic_slice_eager_fallback(input: Annotated[Any, TV_XlaDynamicSlice_T], start_indices: Annotated[Any, TV_XlaDynamicSlice_Tindices], size_indices: Annotated[Any, TV_XlaDynamicSlice_Tindices], name, ctx) -> Annotated[Any, TV_XlaDynamicSlice_T]:
  _attr_T, (input,) = _execute.args_to_matching_eager([input], ctx, [])
  _attr_Tindices, _inputs_Tindices = _execute.args_to_matching_eager([start_indices, size_indices], ctx, [_dtypes.int32, _dtypes.int64, ])
  (start_indices, size_indices) = _inputs_Tindices
  _inputs_flat = [input, start_indices, size_indices]
  _attrs = ("T", _attr_T, "Tindices", _attr_Tindices)
  _result = _execute.execute(b"XlaDynamicSlice", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "XlaDynamicSlice", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_XlaDynamicUpdateSlice_T = TypeVar("TV_XlaDynamicUpdateSlice_T", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float16, _atypes.Float32, _atypes.Float64, _atypes.Float8e4m3fn, _atypes.Float8e5m2, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int4, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.Resource, _atypes.String, _atypes.UInt16, _atypes.UInt32, _atypes.UInt4, _atypes.UInt64, _atypes.UInt8, _atypes.Variant)
TV_XlaDynamicUpdateSlice_Tindices = TypeVar("TV_XlaDynamicUpdateSlice_Tindices", _atypes.Int32, _atypes.Int64)

@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('xla_dynamic_update_slice')
def xla_dynamic_update_slice(input: Annotated[Any, TV_XlaDynamicUpdateSlice_T], update: Annotated[Any, TV_XlaDynamicUpdateSlice_T], indices: Annotated[Any, TV_XlaDynamicUpdateSlice_Tindices], name=None) -> Annotated[Any, TV_XlaDynamicUpdateSlice_T]:
  r"""Wraps the XLA DynamicUpdateSlice operator, documented at

   https://www.tensorflow.org/performance/xla/operation_semantics#dynamicupdateslice
  .

  XlaDynamicUpdateSlice generates a result which is the value of the `input`
  operand, with a slice update overwritten at `indices`. The shape of `update`
  determines the shape of the sub-array of the result which is updated. The shape
  of indices must be rank == 1, with dimension size equal to the rank of `input`.

  Handling of out-of-bounds slice indices is implementation-defined.

  Args:
    input: A `Tensor`. A `Tensor` of type T.
    update: A `Tensor`. Must have the same type as `input`.
      A `Tensor` of type T. Same rank as `input`.
    indices: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      A vector of indices into `input`. Must have length equal to the rank of
      `input`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`. A `Tensor` of type T.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "XlaDynamicUpdateSlice", name, input, update, indices)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_xla_dynamic_update_slice(
          (input, update, indices, name,), None)
      if _result is not NotImplemented:
        return _result
      return xla_dynamic_update_slice_eager_fallback(
          input, update, indices, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            xla_dynamic_update_slice, (), dict(input=input, update=update,
                                               indices=indices, name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_xla_dynamic_update_slice(
        (input, update, indices, name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "XlaDynamicUpdateSlice", input=input, update=update, indices=indices,
                                 name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          xla_dynamic_update_slice, (), dict(input=input, update=update,
                                             indices=indices, name=name)
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
        "XlaDynamicUpdateSlice", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

XlaDynamicUpdateSlice = tf_export("raw_ops.XlaDynamicUpdateSlice")(_ops.to_raw_op(xla_dynamic_update_slice))
_dispatcher_for_xla_dynamic_update_slice = xla_dynamic_update_slice._tf_type_based_dispatcher.Dispatch


def xla_dynamic_update_slice_eager_fallback(input: Annotated[Any, TV_XlaDynamicUpdateSlice_T], update: Annotated[Any, TV_XlaDynamicUpdateSlice_T], indices: Annotated[Any, TV_XlaDynamicUpdateSlice_Tindices], name, ctx) -> Annotated[Any, TV_XlaDynamicUpdateSlice_T]:
  _attr_T, _inputs_T = _execute.args_to_matching_eager([input, update], ctx, [])
  (input, update) = _inputs_T
  _attr_Tindices, (indices,) = _execute.args_to_matching_eager([indices], ctx, [_dtypes.int32, _dtypes.int64, ])
  _inputs_flat = [input, update, indices]
  _attrs = ("T", _attr_T, "Tindices", _attr_Tindices)
  _result = _execute.execute(b"XlaDynamicUpdateSlice", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "XlaDynamicUpdateSlice", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_XlaEinsum_T = TypeVar("TV_XlaEinsum_T", _atypes.BFloat16, _atypes.Complex64, _atypes.Float32)

@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('xla_einsum')
def xla_einsum(a: Annotated[Any, TV_XlaEinsum_T], b: Annotated[Any, TV_XlaEinsum_T], equation: str, name=None) -> Annotated[Any, TV_XlaEinsum_T]:
  r"""An op which supports basic einsum op with 2 inputs and 1 output.

  This op has better TPU performance since it doesn't have explicitly reshape and
  transpose operations as tf.einsum does.

  Args:
    a: A `Tensor`. Must be one of the following types: `complex64`, `bfloat16`, `float32`.
    b: A `Tensor`. Must have the same type as `a`.
    equation: A `string`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `a`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "XlaEinsum", name, a, b, "equation", equation)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_xla_einsum(
          (a, b, equation, name,), None)
      if _result is not NotImplemented:
        return _result
      return xla_einsum_eager_fallback(
          a, b, equation=equation, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            xla_einsum, (), dict(a=a, b=b, equation=equation, name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_xla_einsum(
        (a, b, equation, name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  equation = _execute.make_str(equation, "equation")
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "XlaEinsum", a=a, b=b, equation=equation, name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          xla_einsum, (), dict(a=a, b=b, equation=equation, name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("equation", _op.get_attr("equation"), "T",
              _op._get_attr_type("T"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "XlaEinsum", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

XlaEinsum = tf_export("raw_ops.XlaEinsum")(_ops.to_raw_op(xla_einsum))
_dispatcher_for_xla_einsum = xla_einsum._tf_type_based_dispatcher.Dispatch


def xla_einsum_eager_fallback(a: Annotated[Any, TV_XlaEinsum_T], b: Annotated[Any, TV_XlaEinsum_T], equation: str, name, ctx) -> Annotated[Any, TV_XlaEinsum_T]:
  equation = _execute.make_str(equation, "equation")
  _attr_T, _inputs_T = _execute.args_to_matching_eager([a, b], ctx, [_dtypes.complex64, _dtypes.bfloat16, _dtypes.float32, ])
  (a, b) = _inputs_T
  _inputs_flat = [a, b]
  _attrs = ("equation", equation, "T", _attr_T)
  _result = _execute.execute(b"XlaEinsum", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "XlaEinsum", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_XlaGather_T = TypeVar("TV_XlaGather_T", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float32, _atypes.Float64, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.UInt16, _atypes.UInt32, _atypes.UInt64, _atypes.UInt8)
TV_XlaGather_Tindices = TypeVar("TV_XlaGather_Tindices", _atypes.Int32, _atypes.Int64)

@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('xla_gather')
def xla_gather(operand: Annotated[Any, TV_XlaGather_T], start_indices: Annotated[Any, TV_XlaGather_Tindices], slice_sizes: Annotated[Any, TV_XlaGather_Tindices], dimension_numbers: str, indices_are_sorted: bool, name=None) -> Annotated[Any, TV_XlaGather_T]:
  r"""Wraps the XLA Gather operator documented at

    https://www.tensorflow.org/xla/operation_semantics#gather

  Args:
    operand: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`, `complex64`, `int64`, `qint8`, `quint8`, `qint32`, `bfloat16`, `qint16`, `quint16`, `uint16`, `complex128`, `half`, `uint32`, `uint64`, `bool`.
      The array we're gathering from.
    start_indices: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      Array containing the starting indices of the slices we gather.
    slice_sizes: A `Tensor`. Must have the same type as `start_indices`.
      slice_sizes[i] is the bounds for the slice on dimension i.
    dimension_numbers: A `string`.
      A serialized xla::GatherDimensionNumbers proto.
    indices_are_sorted: A `bool`.
      Boolean indicating if the indices are sorted.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `operand`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "XlaGather", name, operand, start_indices, slice_sizes,
        "dimension_numbers", dimension_numbers, "indices_are_sorted",
        indices_are_sorted)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_xla_gather(
          (operand, start_indices, slice_sizes, dimension_numbers,
          indices_are_sorted, name,), None)
      if _result is not NotImplemented:
        return _result
      return xla_gather_eager_fallback(
          operand, start_indices, slice_sizes,
          dimension_numbers=dimension_numbers,
          indices_are_sorted=indices_are_sorted, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            xla_gather, (), dict(operand=operand, start_indices=start_indices,
                                 slice_sizes=slice_sizes,
                                 dimension_numbers=dimension_numbers,
                                 indices_are_sorted=indices_are_sorted,
                                 name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_xla_gather(
        (operand, start_indices, slice_sizes, dimension_numbers,
        indices_are_sorted, name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  dimension_numbers = _execute.make_str(dimension_numbers, "dimension_numbers")
  indices_are_sorted = _execute.make_bool(indices_are_sorted, "indices_are_sorted")
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "XlaGather", operand=operand, start_indices=start_indices,
                     slice_sizes=slice_sizes,
                     dimension_numbers=dimension_numbers,
                     indices_are_sorted=indices_are_sorted, name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          xla_gather, (), dict(operand=operand, start_indices=start_indices,
                               slice_sizes=slice_sizes,
                               dimension_numbers=dimension_numbers,
                               indices_are_sorted=indices_are_sorted,
                               name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("dimension_numbers", _op.get_attr("dimension_numbers"),
              "indices_are_sorted", _op._get_attr_bool("indices_are_sorted"),
              "T", _op._get_attr_type("T"), "Tindices",
              _op._get_attr_type("Tindices"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "XlaGather", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

XlaGather = tf_export("raw_ops.XlaGather")(_ops.to_raw_op(xla_gather))
_dispatcher_for_xla_gather = xla_gather._tf_type_based_dispatcher.Dispatch


def xla_gather_eager_fallback(operand: Annotated[Any, TV_XlaGather_T], start_indices: Annotated[Any, TV_XlaGather_Tindices], slice_sizes: Annotated[Any, TV_XlaGather_Tindices], dimension_numbers: str, indices_are_sorted: bool, name, ctx) -> Annotated[Any, TV_XlaGather_T]:
  dimension_numbers = _execute.make_str(dimension_numbers, "dimension_numbers")
  indices_are_sorted = _execute.make_bool(indices_are_sorted, "indices_are_sorted")
  _attr_T, (operand,) = _execute.args_to_matching_eager([operand], ctx, [_dtypes.float32, _dtypes.float64, _dtypes.int32, _dtypes.uint8, _dtypes.int16, _dtypes.int8, _dtypes.complex64, _dtypes.int64, _dtypes.qint8, _dtypes.quint8, _dtypes.qint32, _dtypes.bfloat16, _dtypes.qint16, _dtypes.quint16, _dtypes.uint16, _dtypes.complex128, _dtypes.half, _dtypes.uint32, _dtypes.uint64, _dtypes.bool, ])
  _attr_Tindices, _inputs_Tindices = _execute.args_to_matching_eager([start_indices, slice_sizes], ctx, [_dtypes.int32, _dtypes.int64, ])
  (start_indices, slice_sizes) = _inputs_Tindices
  _inputs_flat = [operand, start_indices, slice_sizes]
  _attrs = ("dimension_numbers", dimension_numbers, "indices_are_sorted",
  indices_are_sorted, "T", _attr_T, "Tindices", _attr_Tindices)
  _result = _execute.execute(b"XlaGather", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "XlaGather", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_XlaIf_Tcond = TypeVar("TV_XlaIf_Tcond", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float16, _atypes.Float32, _atypes.Float64, _atypes.Float8e4m3fn, _atypes.Float8e5m2, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int4, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.Resource, _atypes.String, _atypes.UInt16, _atypes.UInt32, _atypes.UInt4, _atypes.UInt64, _atypes.UInt8, _atypes.Variant)

@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('xla_if')
def xla_if(cond: Annotated[Any, TV_XlaIf_Tcond], inputs, then_branch, else_branch, Tout, name=None):
  r"""output = cond ? then_branch(inputs) : else_branch(inputs).

  Args:
    cond: A `Tensor`. A boolean scalar.
    inputs: A list of `Tensor` objects. A list of input tensors.
    then_branch: A function decorated with @Defun.
      A function takes 'inputs' and returns a list of tensors,
      whose types are the same as what else_branch returns.
    else_branch: A function decorated with @Defun.
      A function takes 'inputs' and returns a list of tensors.
      whose types are the same as what then_branch returns.
    Tout: A list of `tf.DTypes`.
    name: A name for the operation (optional).

  Returns:
    A list of `Tensor` objects of type `Tout`.
    A list of tensors returned by either then_branch(inputs) or
    else_branch(inputs). The input shapes of the then_branch and
    else_branch must match.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "XlaIf", name, cond, inputs, "then_branch", then_branch,
        "else_branch", else_branch, "Tout", Tout)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_xla_if(
          (cond, inputs, then_branch, else_branch, Tout, name,), None)
      if _result is not NotImplemented:
        return _result
      return xla_if_eager_fallback(
          cond, inputs, then_branch=then_branch, else_branch=else_branch,
          Tout=Tout, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            xla_if, (), dict(cond=cond, inputs=inputs,
                             then_branch=then_branch, else_branch=else_branch,
                             Tout=Tout, name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_xla_if(
        (cond, inputs, then_branch, else_branch, Tout, name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  if not isinstance(Tout, (list, tuple)):
    raise TypeError(
        "Expected list for 'Tout' argument to "
        "'xla_if' Op, not %r." % Tout)
  Tout = [_execute.make_type(_t, "Tout") for _t in Tout]
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "XlaIf", cond=cond, inputs=inputs, then_branch=then_branch,
                 else_branch=else_branch, Tout=Tout, name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          xla_if, (), dict(cond=cond, inputs=inputs, then_branch=then_branch,
                           else_branch=else_branch, Tout=Tout, name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  _result = _outputs[:]
  if not _result:
    return _op
  if _execute.must_record_gradient():
    _attrs = ("Tcond", _op._get_attr_type("Tcond"), "then_branch",
              _op.get_attr("then_branch"), "else_branch",
              _op.get_attr("else_branch"), "Tin", _op.get_attr("Tin"), "Tout",
              _op.get_attr("Tout"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "XlaIf", _inputs_flat, _attrs, _result)
  return _result

XlaIf = tf_export("raw_ops.XlaIf")(_ops.to_raw_op(xla_if))
_dispatcher_for_xla_if = xla_if._tf_type_based_dispatcher.Dispatch


def xla_if_eager_fallback(cond: Annotated[Any, TV_XlaIf_Tcond], inputs, then_branch, else_branch, Tout, name, ctx):
  if not isinstance(Tout, (list, tuple)):
    raise TypeError(
        "Expected list for 'Tout' argument to "
        "'xla_if' Op, not %r." % Tout)
  Tout = [_execute.make_type(_t, "Tout") for _t in Tout]
  _attr_Tcond, (cond,) = _execute.args_to_matching_eager([cond], ctx, [])
  _attr_Tin, inputs = _execute.convert_to_mixed_eager_tensors(inputs, ctx)
  _inputs_flat = [cond] + list(inputs)
  _attrs = ("Tcond", _attr_Tcond, "then_branch", then_branch, "else_branch",
  else_branch, "Tin", _attr_Tin, "Tout", Tout)
  _result = _execute.execute(b"XlaIf", len(Tout), inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "XlaIf", _inputs_flat, _attrs, _result)
  return _result

_XlaKeyValueSortOutput = collections.namedtuple(
    "XlaKeyValueSort",
    ["sorted_keys", "sorted_values"])


TV_XlaKeyValueSort_K = TypeVar("TV_XlaKeyValueSort_K", _atypes.BFloat16, _atypes.Float32, _atypes.Float64, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int64, _atypes.Int8, _atypes.UInt16, _atypes.UInt32, _atypes.UInt64, _atypes.UInt8)
TV_XlaKeyValueSort_V = TypeVar("TV_XlaKeyValueSort_V", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float16, _atypes.Float32, _atypes.Float64, _atypes.Float8e4m3fn, _atypes.Float8e5m2, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int4, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.Resource, _atypes.String, _atypes.UInt16, _atypes.UInt32, _atypes.UInt4, _atypes.UInt64, _atypes.UInt8, _atypes.Variant)

@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('xla_key_value_sort')
def xla_key_value_sort(keys: Annotated[Any, TV_XlaKeyValueSort_K], values: Annotated[Any, TV_XlaKeyValueSort_V], name=None):
  r"""Wraps the XLA Sort operator, documented at

   https://www.tensorflow.org/performance/xla/operation_semantics#sort
  .

  Sorts a tensor. Currently only sorts in ascending order are supported.

  Args:
    keys: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`, `int64`, `bfloat16`, `uint16`, `half`, `uint32`, `uint64`.
      A `Tensor` of type K.
    values: A `Tensor`. A `Tensor` of type V.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (sorted_keys, sorted_values).

    sorted_keys: A `Tensor`. Has the same type as `keys`. A `Tensor` of type K.
    sorted_values: A `Tensor`. Has the same type as `values`. A `Tensor` of type V.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "XlaKeyValueSort", name, keys, values)
      _result = _XlaKeyValueSortOutput._make(_result)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_xla_key_value_sort(
          (keys, values, name,), None)
      if _result is not NotImplemented:
        return _result
      return xla_key_value_sort_eager_fallback(
          keys, values, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            xla_key_value_sort, (), dict(keys=keys, values=values, name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_xla_key_value_sort(
        (keys, values, name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "XlaKeyValueSort", keys=keys, values=values, name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          xla_key_value_sort, (), dict(keys=keys, values=values, name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("K", _op._get_attr_type("K"), "V", _op._get_attr_type("V"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "XlaKeyValueSort", _inputs_flat, _attrs, _result)
  _result = _XlaKeyValueSortOutput._make(_result)
  return _result

XlaKeyValueSort = tf_export("raw_ops.XlaKeyValueSort")(_ops.to_raw_op(xla_key_value_sort))
_dispatcher_for_xla_key_value_sort = xla_key_value_sort._tf_type_based_dispatcher.Dispatch


def xla_key_value_sort_eager_fallback(keys: Annotated[Any, TV_XlaKeyValueSort_K], values: Annotated[Any, TV_XlaKeyValueSort_V], name, ctx):
  _attr_K, (keys,) = _execute.args_to_matching_eager([keys], ctx, [_dtypes.float32, _dtypes.float64, _dtypes.int32, _dtypes.uint8, _dtypes.int16, _dtypes.int8, _dtypes.int64, _dtypes.bfloat16, _dtypes.uint16, _dtypes.half, _dtypes.uint32, _dtypes.uint64, ])
  _attr_V, (values,) = _execute.args_to_matching_eager([values], ctx, [])
  _inputs_flat = [keys, values]
  _attrs = ("K", _attr_K, "V", _attr_V)
  _result = _execute.execute(b"XlaKeyValueSort", 2, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "XlaKeyValueSort", _inputs_flat, _attrs, _result)
  _result = _XlaKeyValueSortOutput._make(_result)
  return _result


@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('xla_optimization_barrier')
def xla_optimization_barrier(input, name=None):
  r"""Wraps the XLA OptimizationBarrier operator.

  Documented at https://www.tensorflow.org/xla/operation_semantics#optimizationbarrier.

  Args:
    input: A list of `Tensor` objects. A Tuple of Arrays of any type.
    name: A name for the operation (optional).

  Returns:
    A list of `Tensor` objects. Has the same type as `input`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "XlaOptimizationBarrier", name, input)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_xla_optimization_barrier(
          (input, name,), None)
      if _result is not NotImplemented:
        return _result
      return xla_optimization_barrier_eager_fallback(
          input, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            xla_optimization_barrier, (), dict(input=input, name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_xla_optimization_barrier(
        (input, name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "XlaOptimizationBarrier", input=input, name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          xla_optimization_barrier, (), dict(input=input, name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op.get_attr("T"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "XlaOptimizationBarrier", _inputs_flat, _attrs, _result)
  return _result

XlaOptimizationBarrier = tf_export("raw_ops.XlaOptimizationBarrier")(_ops.to_raw_op(xla_optimization_barrier))
_dispatcher_for_xla_optimization_barrier = xla_optimization_barrier._tf_type_based_dispatcher.Dispatch


def xla_optimization_barrier_eager_fallback(input, name, ctx):
  _attr_T, input = _execute.convert_to_mixed_eager_tensors(input, ctx)
  _inputs_flat = list(input)
  _attrs = ("T", _attr_T)
  _result = _execute.execute(b"XlaOptimizationBarrier", len(input),
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "XlaOptimizationBarrier", _inputs_flat, _attrs, _result)
  return _result


TV_XlaPad_T = TypeVar("TV_XlaPad_T", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float16, _atypes.Float32, _atypes.Float64, _atypes.Float8e4m3fn, _atypes.Float8e5m2, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int4, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.Resource, _atypes.String, _atypes.UInt16, _atypes.UInt32, _atypes.UInt4, _atypes.UInt64, _atypes.UInt8, _atypes.Variant)
TV_XlaPad_Tindices = TypeVar("TV_XlaPad_Tindices", _atypes.Int32, _atypes.Int64)

@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('xla_pad')
def xla_pad(input: Annotated[Any, TV_XlaPad_T], padding_value: Annotated[Any, TV_XlaPad_T], padding_low: Annotated[Any, TV_XlaPad_Tindices], padding_high: Annotated[Any, TV_XlaPad_Tindices], padding_interior: Annotated[Any, TV_XlaPad_Tindices], name=None) -> Annotated[Any, TV_XlaPad_T]:
  r"""Wraps the XLA Pad operator, documented at

   https://www.tensorflow.org/performance/xla/operation_semantics#pad
  .

  Args:
    input: A `Tensor`. A `Tensor` of type T.
    padding_value: A `Tensor`. Must have the same type as `input`.
      A scalar `Tensor` of type T.
    padding_low: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      the padding to apply at the start of each input dimensions. Must
      be a compile-time constant 1D tensor of length equal to rank of input.
    padding_high: A `Tensor`. Must have the same type as `padding_low`.
      the padding to apply at the end of each input dimension. Must
      be a compile-time constant 1D tensor of length equal to rank of input.
    padding_interior: A `Tensor`. Must have the same type as `padding_low`.
      the padding to apply between each input element. Must
      be a compile-time constant 1D tensor of length equal to rank of input,
      containing only non-negative values.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`. A `Tensor` of type T.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "XlaPad", name, input, padding_value, padding_low, padding_high,
        padding_interior)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_xla_pad(
          (input, padding_value, padding_low, padding_high, padding_interior,
          name,), None)
      if _result is not NotImplemented:
        return _result
      return xla_pad_eager_fallback(
          input, padding_value, padding_low, padding_high, padding_interior,
          name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            xla_pad, (), dict(input=input, padding_value=padding_value,
                              padding_low=padding_low,
                              padding_high=padding_high,
                              padding_interior=padding_interior, name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_xla_pad(
        (input, padding_value, padding_low, padding_high, padding_interior,
        name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "XlaPad", input=input, padding_value=padding_value,
                  padding_low=padding_low, padding_high=padding_high,
                  padding_interior=padding_interior, name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          xla_pad, (), dict(input=input, padding_value=padding_value,
                            padding_low=padding_low,
                            padding_high=padding_high,
                            padding_interior=padding_interior, name=name)
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
        "XlaPad", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

XlaPad = tf_export("raw_ops.XlaPad")(_ops.to_raw_op(xla_pad))
_dispatcher_for_xla_pad = xla_pad._tf_type_based_dispatcher.Dispatch


def xla_pad_eager_fallback(input: Annotated[Any, TV_XlaPad_T], padding_value: Annotated[Any, TV_XlaPad_T], padding_low: Annotated[Any, TV_XlaPad_Tindices], padding_high: Annotated[Any, TV_XlaPad_Tindices], padding_interior: Annotated[Any, TV_XlaPad_Tindices], name, ctx) -> Annotated[Any, TV_XlaPad_T]:
  _attr_T, _inputs_T = _execute.args_to_matching_eager([input, padding_value], ctx, [])
  (input, padding_value) = _inputs_T
  _attr_Tindices, _inputs_Tindices = _execute.args_to_matching_eager([padding_low, padding_high, padding_interior], ctx, [_dtypes.int32, _dtypes.int64, ])
  (padding_low, padding_high, padding_interior) = _inputs_Tindices
  _inputs_flat = [input, padding_value, padding_low, padding_high, padding_interior]
  _attrs = ("T", _attr_T, "Tindices", _attr_Tindices)
  _result = _execute.execute(b"XlaPad", 1, inputs=_inputs_flat, attrs=_attrs,
                             ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "XlaPad", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_XlaRecv_dtype = TypeVar("TV_XlaRecv_dtype", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float16, _atypes.Float32, _atypes.Float64, _atypes.Float8e4m3fn, _atypes.Float8e5m2, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int4, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.Resource, _atypes.String, _atypes.UInt16, _atypes.UInt32, _atypes.UInt4, _atypes.UInt64, _atypes.UInt8, _atypes.Variant)

@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('xla_recv')
def xla_recv(dtype: TV_XlaRecv_dtype, tensor_name: str, shape, name=None) -> Annotated[Any, TV_XlaRecv_dtype]:
  r"""Receives the named tensor from another XLA computation. Wraps the XLA Recv

  operator documented at
   https://www.tensorflow.org/performance/xla/operation_semantics#recv .

  Args:
    dtype: A `tf.DType`. The type of the tensor.
    tensor_name: A `string`. A string key that identifies the channel.
    shape: A `tf.TensorShape` or list of `ints`. The shape of the tensor.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `dtype`. The tensor to receive.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "XlaRecv", name, "dtype", dtype, "tensor_name", tensor_name,
        "shape", shape)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_xla_recv(
          (dtype, tensor_name, shape, name,), None)
      if _result is not NotImplemented:
        return _result
      return xla_recv_eager_fallback(
          dtype=dtype, tensor_name=tensor_name, shape=shape, name=name,
          ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            xla_recv, (), dict(dtype=dtype, tensor_name=tensor_name,
                               shape=shape, name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_xla_recv(
        (dtype, tensor_name, shape, name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  dtype = _execute.make_type(dtype, "dtype")
  tensor_name = _execute.make_str(tensor_name, "tensor_name")
  shape = _execute.make_shape(shape, "shape")
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "XlaRecv", dtype=dtype, tensor_name=tensor_name, shape=shape,
                   name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          xla_recv, (), dict(dtype=dtype, tensor_name=tensor_name,
                             shape=shape, name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("dtype", _op._get_attr_type("dtype"), "tensor_name",
              _op.get_attr("tensor_name"), "shape", _op.get_attr("shape"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "XlaRecv", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

XlaRecv = tf_export("raw_ops.XlaRecv")(_ops.to_raw_op(xla_recv))
_dispatcher_for_xla_recv = xla_recv._tf_type_based_dispatcher.Dispatch


def xla_recv_eager_fallback(dtype: TV_XlaRecv_dtype, tensor_name: str, shape, name, ctx) -> Annotated[Any, TV_XlaRecv_dtype]:
  dtype = _execute.make_type(dtype, "dtype")
  tensor_name = _execute.make_str(tensor_name, "tensor_name")
  shape = _execute.make_shape(shape, "shape")
  _inputs_flat = []
  _attrs = ("dtype", dtype, "tensor_name", tensor_name, "shape", shape)
  _result = _execute.execute(b"XlaRecv", 1, inputs=_inputs_flat, attrs=_attrs,
                             ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "XlaRecv", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_XlaReduce_T = TypeVar("TV_XlaReduce_T", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float32, _atypes.Float64, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.UInt16, _atypes.UInt32, _atypes.UInt64, _atypes.UInt8)

@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('xla_reduce')
def xla_reduce(input: Annotated[Any, TV_XlaReduce_T], init_value: Annotated[Any, TV_XlaReduce_T], dimensions_to_reduce, reducer, name=None) -> Annotated[Any, TV_XlaReduce_T]:
  r"""Wraps the XLA Reduce operator, documented at

   https://www.tensorflow.org/performance/xla/operation_semantics#reduce .

  Args:
    input: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`, `complex64`, `int64`, `qint8`, `quint8`, `qint32`, `bfloat16`, `qint16`, `quint16`, `uint16`, `complex128`, `half`, `uint32`, `uint64`, `bool`.
      the input tensor
    init_value: A `Tensor`. Must have the same type as `input`.
      a scalar representing the initial value for the reduction
    dimensions_to_reduce: A list of `ints`.
      dimension numbers over which to reduce
    reducer: A function decorated with @Defun. a reducer function to apply
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "XlaReduce", name, input, init_value, "dimensions_to_reduce",
        dimensions_to_reduce, "reducer", reducer)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_xla_reduce(
          (input, init_value, dimensions_to_reduce, reducer, name,), None)
      if _result is not NotImplemented:
        return _result
      return xla_reduce_eager_fallback(
          input, init_value, dimensions_to_reduce=dimensions_to_reduce,
          reducer=reducer, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            xla_reduce, (), dict(input=input, init_value=init_value,
                                 dimensions_to_reduce=dimensions_to_reduce,
                                 reducer=reducer, name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_xla_reduce(
        (input, init_value, dimensions_to_reduce, reducer, name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  if not isinstance(dimensions_to_reduce, (list, tuple)):
    raise TypeError(
        "Expected list for 'dimensions_to_reduce' argument to "
        "'xla_reduce' Op, not %r." % dimensions_to_reduce)
  dimensions_to_reduce = [_execute.make_int(_i, "dimensions_to_reduce") for _i in dimensions_to_reduce]
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "XlaReduce", input=input, init_value=init_value,
                     dimensions_to_reduce=dimensions_to_reduce,
                     reducer=reducer, name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          xla_reduce, (), dict(input=input, init_value=init_value,
                               dimensions_to_reduce=dimensions_to_reduce,
                               reducer=reducer, name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"), "dimensions_to_reduce",
              _op.get_attr("dimensions_to_reduce"), "reducer",
              _op.get_attr("reducer"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "XlaReduce", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

XlaReduce = tf_export("raw_ops.XlaReduce")(_ops.to_raw_op(xla_reduce))
_dispatcher_for_xla_reduce = xla_reduce._tf_type_based_dispatcher.Dispatch


def xla_reduce_eager_fallback(input: Annotated[Any, TV_XlaReduce_T], init_value: Annotated[Any, TV_XlaReduce_T], dimensions_to_reduce, reducer, name, ctx) -> Annotated[Any, TV_XlaReduce_T]:
  if not isinstance(dimensions_to_reduce, (list, tuple)):
    raise TypeError(
        "Expected list for 'dimensions_to_reduce' argument to "
        "'xla_reduce' Op, not %r." % dimensions_to_reduce)
  dimensions_to_reduce = [_execute.make_int(_i, "dimensions_to_reduce") for _i in dimensions_to_reduce]
  _attr_T, _inputs_T = _execute.args_to_matching_eager([input, init_value], ctx, [_dtypes.float32, _dtypes.float64, _dtypes.int32, _dtypes.uint8, _dtypes.int16, _dtypes.int8, _dtypes.complex64, _dtypes.int64, _dtypes.qint8, _dtypes.quint8, _dtypes.qint32, _dtypes.bfloat16, _dtypes.qint16, _dtypes.quint16, _dtypes.uint16, _dtypes.complex128, _dtypes.half, _dtypes.uint32, _dtypes.uint64, _dtypes.bool, ])
  (input, init_value) = _inputs_T
  _inputs_flat = [input, init_value]
  _attrs = ("T", _attr_T, "dimensions_to_reduce", dimensions_to_reduce,
  "reducer", reducer)
  _result = _execute.execute(b"XlaReduce", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "XlaReduce", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_XlaReducePrecision_T = TypeVar("TV_XlaReducePrecision_T", _atypes.BFloat16, _atypes.Float32, _atypes.Float64, _atypes.Half)

@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('xla_reduce_precision')
def xla_reduce_precision(operand: Annotated[Any, TV_XlaReducePrecision_T], exponent_bits: int, mantissa_bits: int, name=None) -> Annotated[Any, TV_XlaReducePrecision_T]:
  r"""Wraps the XLA ReducePrecision operator

    documented at https://www.tensorflow.org/xla/operation_semantics#reduceprecision.

  Args:
    operand: A `Tensor`. Must be one of the following types: `bfloat16`, `half`, `float32`, `float64`.
      array of floating-point type.
    exponent_bits: An `int`. number of exponent bits in lower-precision format
    mantissa_bits: An `int`. number of mantissa bits in lower-precision format
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `operand`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "XlaReducePrecision", name, operand, "exponent_bits",
        exponent_bits, "mantissa_bits", mantissa_bits)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_xla_reduce_precision(
          (operand, exponent_bits, mantissa_bits, name,), None)
      if _result is not NotImplemented:
        return _result
      return xla_reduce_precision_eager_fallback(
          operand, exponent_bits=exponent_bits, mantissa_bits=mantissa_bits,
          name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            xla_reduce_precision, (), dict(operand=operand,
                                           exponent_bits=exponent_bits,
                                           mantissa_bits=mantissa_bits,
                                           name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_xla_reduce_precision(
        (operand, exponent_bits, mantissa_bits, name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  exponent_bits = _execute.make_int(exponent_bits, "exponent_bits")
  mantissa_bits = _execute.make_int(mantissa_bits, "mantissa_bits")
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "XlaReducePrecision", operand=operand, exponent_bits=exponent_bits,
                              mantissa_bits=mantissa_bits, name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          xla_reduce_precision, (), dict(operand=operand,
                                         exponent_bits=exponent_bits,
                                         mantissa_bits=mantissa_bits,
                                         name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"), "exponent_bits",
              _op._get_attr_int("exponent_bits"), "mantissa_bits",
              _op._get_attr_int("mantissa_bits"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "XlaReducePrecision", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

XlaReducePrecision = tf_export("raw_ops.XlaReducePrecision")(_ops.to_raw_op(xla_reduce_precision))
_dispatcher_for_xla_reduce_precision = xla_reduce_precision._tf_type_based_dispatcher.Dispatch


def xla_reduce_precision_eager_fallback(operand: Annotated[Any, TV_XlaReducePrecision_T], exponent_bits: int, mantissa_bits: int, name, ctx) -> Annotated[Any, TV_XlaReducePrecision_T]:
  exponent_bits = _execute.make_int(exponent_bits, "exponent_bits")
  mantissa_bits = _execute.make_int(mantissa_bits, "mantissa_bits")
  _attr_T, (operand,) = _execute.args_to_matching_eager([operand], ctx, [_dtypes.bfloat16, _dtypes.half, _dtypes.float32, _dtypes.float64, ])
  _inputs_flat = [operand]
  _attrs = ("T", _attr_T, "exponent_bits", exponent_bits, "mantissa_bits",
  mantissa_bits)
  _result = _execute.execute(b"XlaReducePrecision", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "XlaReducePrecision", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_XlaReduceScatter_T = TypeVar("TV_XlaReduceScatter_T", _atypes.BFloat16, _atypes.Float32, _atypes.Half, _atypes.Int32, _atypes.UInt32)

@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('xla_reduce_scatter')
def xla_reduce_scatter(input: Annotated[Any, TV_XlaReduceScatter_T], group_assignment: Annotated[Any, _atypes.Int32], scatter_dimension: Annotated[Any, _atypes.Int32], reduce_op: str, name=None) -> Annotated[Any, TV_XlaReduceScatter_T]:
  r"""Wraps the XLA ReduceScatter operator

    documented at https://www.tensorflow.org/xla/operation_semantics#reducescatter.

  Args:
    input: A `Tensor`. Must be one of the following types: `half`, `bfloat16`, `float32`, `int32`, `uint32`.
      Array or a non-empty tuple of arrays to reduce across replicas.
    group_assignment: A `Tensor` of type `int32`.
      Groups between which the reductions are performed.
    scatter_dimension: A `Tensor` of type `int32`. Dimension to scatter.
    reduce_op: A `string` from: `"Min", "Max", "Mul", "Add", "Mean"`.
      Reduction computation.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "XlaReduceScatter", name, input, group_assignment,
        scatter_dimension, "reduce_op", reduce_op)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_xla_reduce_scatter(
          (input, group_assignment, scatter_dimension, reduce_op, name,), None)
      if _result is not NotImplemented:
        return _result
      return xla_reduce_scatter_eager_fallback(
          input, group_assignment, scatter_dimension, reduce_op=reduce_op,
          name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            xla_reduce_scatter, (), dict(input=input,
                                         group_assignment=group_assignment,
                                         scatter_dimension=scatter_dimension,
                                         reduce_op=reduce_op, name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_xla_reduce_scatter(
        (input, group_assignment, scatter_dimension, reduce_op, name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  reduce_op = _execute.make_str(reduce_op, "reduce_op")
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "XlaReduceScatter", input=input, group_assignment=group_assignment,
                            scatter_dimension=scatter_dimension,
                            reduce_op=reduce_op, name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          xla_reduce_scatter, (), dict(input=input,
                                       group_assignment=group_assignment,
                                       scatter_dimension=scatter_dimension,
                                       reduce_op=reduce_op, name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"), "reduce_op",
              _op.get_attr("reduce_op"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "XlaReduceScatter", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

XlaReduceScatter = tf_export("raw_ops.XlaReduceScatter")(_ops.to_raw_op(xla_reduce_scatter))
_dispatcher_for_xla_reduce_scatter = xla_reduce_scatter._tf_type_based_dispatcher.Dispatch


def xla_reduce_scatter_eager_fallback(input: Annotated[Any, TV_XlaReduceScatter_T], group_assignment: Annotated[Any, _atypes.Int32], scatter_dimension: Annotated[Any, _atypes.Int32], reduce_op: str, name, ctx) -> Annotated[Any, TV_XlaReduceScatter_T]:
  reduce_op = _execute.make_str(reduce_op, "reduce_op")
  _attr_T, (input,) = _execute.args_to_matching_eager([input], ctx, [_dtypes.half, _dtypes.bfloat16, _dtypes.float32, _dtypes.int32, _dtypes.uint32, ])
  group_assignment = _ops.convert_to_tensor(group_assignment, _dtypes.int32)
  scatter_dimension = _ops.convert_to_tensor(scatter_dimension, _dtypes.int32)
  _inputs_flat = [input, group_assignment, scatter_dimension]
  _attrs = ("T", _attr_T, "reduce_op", reduce_op)
  _result = _execute.execute(b"XlaReduceScatter", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "XlaReduceScatter", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_XlaReduceWindow_T = TypeVar("TV_XlaReduceWindow_T", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float32, _atypes.Float64, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.UInt16, _atypes.UInt32, _atypes.UInt64, _atypes.UInt8)
TV_XlaReduceWindow_Tindices = TypeVar("TV_XlaReduceWindow_Tindices", _atypes.Int32, _atypes.Int64)

@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('xla_reduce_window')
def xla_reduce_window(input: Annotated[Any, TV_XlaReduceWindow_T], init_value: Annotated[Any, TV_XlaReduceWindow_T], window_dimensions: Annotated[Any, TV_XlaReduceWindow_Tindices], window_strides: Annotated[Any, TV_XlaReduceWindow_Tindices], base_dilations: Annotated[Any, TV_XlaReduceWindow_Tindices], window_dilations: Annotated[Any, TV_XlaReduceWindow_Tindices], padding: Annotated[Any, TV_XlaReduceWindow_Tindices], computation, name=None) -> Annotated[Any, TV_XlaReduceWindow_T]:
  r"""Wraps the XLA ReduceWindow operator, documented at

   https://www.tensorflow.org/performance/xla/operation_semantics#reducewindow .

  Args:
    input: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`, `complex64`, `int64`, `qint8`, `quint8`, `qint32`, `bfloat16`, `qint16`, `quint16`, `uint16`, `complex128`, `half`, `uint32`, `uint64`, `bool`.
      the input tensor
    init_value: A `Tensor`. Must have the same type as `input`.
      a scalar representing the initial value for the reduction
    window_dimensions: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      the shape of the window
    window_strides: A `Tensor`. Must have the same type as `window_dimensions`.
      the inter-window strides
    base_dilations: A `Tensor`. Must have the same type as `window_dimensions`.
    window_dilations: A `Tensor`. Must have the same type as `window_dimensions`.
    padding: A `Tensor`. Must have the same type as `window_dimensions`.
      the padding to apply at the start and end of each input dimensions
    computation: A function decorated with @Defun. a reducer function to apply
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "XlaReduceWindow", name, input, init_value, window_dimensions,
        window_strides, base_dilations, window_dilations, padding,
        "computation", computation)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_xla_reduce_window(
          (input, init_value, window_dimensions, window_strides,
          base_dilations, window_dilations, padding, computation, name,), None)
      if _result is not NotImplemented:
        return _result
      return xla_reduce_window_eager_fallback(
          input, init_value, window_dimensions, window_strides,
          base_dilations, window_dilations, padding, computation=computation,
          name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            xla_reduce_window, (), dict(input=input, init_value=init_value,
                                        window_dimensions=window_dimensions,
                                        window_strides=window_strides,
                                        base_dilations=base_dilations,
                                        window_dilations=window_dilations,
                                        padding=padding,
                                        computation=computation, name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_xla_reduce_window(
        (input, init_value, window_dimensions, window_strides, base_dilations,
        window_dilations, padding, computation, name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "XlaReduceWindow", input=input, init_value=init_value,
                           window_dimensions=window_dimensions,
                           window_strides=window_strides,
                           base_dilations=base_dilations,
                           window_dilations=window_dilations, padding=padding,
                           computation=computation, name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          xla_reduce_window, (), dict(input=input, init_value=init_value,
                                      window_dimensions=window_dimensions,
                                      window_strides=window_strides,
                                      base_dilations=base_dilations,
                                      window_dilations=window_dilations,
                                      padding=padding,
                                      computation=computation, name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"), "Tindices",
              _op._get_attr_type("Tindices"), "computation",
              _op.get_attr("computation"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "XlaReduceWindow", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

XlaReduceWindow = tf_export("raw_ops.XlaReduceWindow")(_ops.to_raw_op(xla_reduce_window))
_dispatcher_for_xla_reduce_window = xla_reduce_window._tf_type_based_dispatcher.Dispatch


def xla_reduce_window_eager_fallback(input: Annotated[Any, TV_XlaReduceWindow_T], init_value: Annotated[Any, TV_XlaReduceWindow_T], window_dimensions: Annotated[Any, TV_XlaReduceWindow_Tindices], window_strides: Annotated[Any, TV_XlaReduceWindow_Tindices], base_dilations: Annotated[Any, TV_XlaReduceWindow_Tindices], window_dilations: Annotated[Any, TV_XlaReduceWindow_Tindices], padding: Annotated[Any, TV_XlaReduceWindow_Tindices], computation, name, ctx) -> Annotated[Any, TV_XlaReduceWindow_T]:
  _attr_T, _inputs_T = _execute.args_to_matching_eager([input, init_value], ctx, [_dtypes.float32, _dtypes.float64, _dtypes.int32, _dtypes.uint8, _dtypes.int16, _dtypes.int8, _dtypes.complex64, _dtypes.int64, _dtypes.qint8, _dtypes.quint8, _dtypes.qint32, _dtypes.bfloat16, _dtypes.qint16, _dtypes.quint16, _dtypes.uint16, _dtypes.complex128, _dtypes.half, _dtypes.uint32, _dtypes.uint64, _dtypes.bool, ])
  (input, init_value) = _inputs_T
  _attr_Tindices, _inputs_Tindices = _execute.args_to_matching_eager([window_dimensions, window_strides, base_dilations, window_dilations, padding], ctx, [_dtypes.int32, _dtypes.int64, ])
  (window_dimensions, window_strides, base_dilations, window_dilations, padding) = _inputs_Tindices
  _inputs_flat = [input, init_value, window_dimensions, window_strides, base_dilations, window_dilations, padding]
  _attrs = ("T", _attr_T, "Tindices", _attr_Tindices, "computation",
  computation)
  _result = _execute.execute(b"XlaReduceWindow", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "XlaReduceWindow", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_XlaRemoveDynamicDimensionSize_T = TypeVar("TV_XlaRemoveDynamicDimensionSize_T", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float16, _atypes.Float32, _atypes.Float64, _atypes.Float8e4m3fn, _atypes.Float8e5m2, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int4, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.Resource, _atypes.String, _atypes.UInt16, _atypes.UInt32, _atypes.UInt4, _atypes.UInt64, _atypes.UInt8, _atypes.Variant)

@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('xla_remove_dynamic_dimension_size')
def xla_remove_dynamic_dimension_size(input: Annotated[Any, TV_XlaRemoveDynamicDimensionSize_T], dim_index: Annotated[Any, _atypes.Int32], name=None) -> Annotated[Any, TV_XlaRemoveDynamicDimensionSize_T]:
  r"""Inverse of XlaSetDynamicDimensionSize.

  Make an xla bounded dynamic dimension into a static dimension. The bound of the
  size of dimension `dim_index` becomes the static dimension size.

  Args:
    input: A `Tensor`.
    dim_index: A `Tensor` of type `int32`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "XlaRemoveDynamicDimensionSize", name, input, dim_index)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_xla_remove_dynamic_dimension_size(
          (input, dim_index, name,), None)
      if _result is not NotImplemented:
        return _result
      return xla_remove_dynamic_dimension_size_eager_fallback(
          input, dim_index, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            xla_remove_dynamic_dimension_size, (), dict(input=input,
                                                        dim_index=dim_index,
                                                        name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_xla_remove_dynamic_dimension_size(
        (input, dim_index, name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "XlaRemoveDynamicDimensionSize", input=input, dim_index=dim_index,
                                         name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          xla_remove_dynamic_dimension_size, (), dict(input=input,
                                                      dim_index=dim_index,
                                                      name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "XlaRemoveDynamicDimensionSize", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

XlaRemoveDynamicDimensionSize = tf_export("raw_ops.XlaRemoveDynamicDimensionSize")(_ops.to_raw_op(xla_remove_dynamic_dimension_size))
_dispatcher_for_xla_remove_dynamic_dimension_size = xla_remove_dynamic_dimension_size._tf_type_based_dispatcher.Dispatch


def xla_remove_dynamic_dimension_size_eager_fallback(input: Annotated[Any, TV_XlaRemoveDynamicDimensionSize_T], dim_index: Annotated[Any, _atypes.Int32], name, ctx) -> Annotated[Any, TV_XlaRemoveDynamicDimensionSize_T]:
  _attr_T, (input,) = _execute.args_to_matching_eager([input], ctx, [])
  dim_index = _ops.convert_to_tensor(dim_index, _dtypes.int32)
  _inputs_flat = [input, dim_index]
  _attrs = ("T", _attr_T)
  _result = _execute.execute(b"XlaRemoveDynamicDimensionSize", 1,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "XlaRemoveDynamicDimensionSize", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('xla_replica_id')
def xla_replica_id(name=None) -> Annotated[Any, _atypes.Int32]:
  r"""Replica ID.

  Args:
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `int32`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "XlaReplicaId", name)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_xla_replica_id(
          (name,), None)
      if _result is not NotImplemented:
        return _result
      return xla_replica_id_eager_fallback(
          name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            xla_replica_id, (), dict(name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_xla_replica_id(
        (name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "XlaReplicaId", name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          xla_replica_id, (), dict(name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ()
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "XlaReplicaId", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

XlaReplicaId = tf_export("raw_ops.XlaReplicaId")(_ops.to_raw_op(xla_replica_id))
_dispatcher_for_xla_replica_id = xla_replica_id._tf_type_based_dispatcher.Dispatch


def xla_replica_id_eager_fallback(name, ctx) -> Annotated[Any, _atypes.Int32]:
  _inputs_flat = []
  _attrs = None
  _result = _execute.execute(b"XlaReplicaId", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "XlaReplicaId", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

_XlaRngBitGeneratorOutput = collections.namedtuple(
    "XlaRngBitGenerator",
    ["output_key", "output"])


TV_XlaRngBitGenerator_dtype = TypeVar("TV_XlaRngBitGenerator_dtype", _atypes.Int32, _atypes.Int64, _atypes.Int8, _atypes.UInt32, _atypes.UInt64, _atypes.UInt8)
TV_XlaRngBitGenerator_Tshape = TypeVar("TV_XlaRngBitGenerator_Tshape", _atypes.Int32, _atypes.Int64)

@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('xla_rng_bit_generator')
def xla_rng_bit_generator(algorithm: Annotated[Any, _atypes.Int32], initial_state: Annotated[Any, _atypes.UInt64], shape: Annotated[Any, TV_XlaRngBitGenerator_Tshape], dtype:TV_XlaRngBitGenerator_dtype=_dtypes.uint64, name=None):
  r"""Stateless PRNG bit generator.

  Wraps the XLA RngBitGenerator operator, documented at
   https://www.tensorflow.org/performance/xla/operation_semantics#rngbitgenerator.

  Args:
    algorithm: A `Tensor` of type `int32`. The PRNG algorithm to use, one of
      tf.random.Algorithm.{PHILOX, THREEFRY, AUTO_SELECT}.
    initial_state: A `Tensor` of type `uint64`.
      Initial state for the PRNG algorithm. For THREEFRY, it should be
      a u64[2] and for PHILOX a u64[3].
    shape: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      The output shape of the generated data.
    dtype: An optional `tf.DType` from: `tf.uint8, tf.int8, tf.int32, tf.int64, tf.uint32, tf.uint64`. Defaults to `tf.uint64`.
      The type of the tensor.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (output_key, output).

    output_key: A `Tensor` of type `uint64`.
    output: A `Tensor` of type `dtype`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "XlaRngBitGenerator", name, algorithm, initial_state, shape,
        "dtype", dtype)
      _result = _XlaRngBitGeneratorOutput._make(_result)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_xla_rng_bit_generator(
          (algorithm, initial_state, shape, dtype, name,), None)
      if _result is not NotImplemented:
        return _result
      return xla_rng_bit_generator_eager_fallback(
          algorithm, initial_state, shape, dtype=dtype, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            xla_rng_bit_generator, (), dict(algorithm=algorithm,
                                            initial_state=initial_state,
                                            shape=shape, dtype=dtype,
                                            name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_xla_rng_bit_generator(
        (algorithm, initial_state, shape, dtype, name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  if dtype is None:
    dtype = _dtypes.uint64
  dtype = _execute.make_type(dtype, "dtype")
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "XlaRngBitGenerator", algorithm=algorithm,
                              initial_state=initial_state, shape=shape,
                              dtype=dtype, name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          xla_rng_bit_generator, (), dict(algorithm=algorithm,
                                          initial_state=initial_state,
                                          shape=shape, dtype=dtype, name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("dtype", _op._get_attr_type("dtype"), "Tshape",
              _op._get_attr_type("Tshape"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "XlaRngBitGenerator", _inputs_flat, _attrs, _result)
  _result = _XlaRngBitGeneratorOutput._make(_result)
  return _result

XlaRngBitGenerator = tf_export("raw_ops.XlaRngBitGenerator")(_ops.to_raw_op(xla_rng_bit_generator))
_dispatcher_for_xla_rng_bit_generator = xla_rng_bit_generator._tf_type_based_dispatcher.Dispatch


def xla_rng_bit_generator_eager_fallback(algorithm: Annotated[Any, _atypes.Int32], initial_state: Annotated[Any, _atypes.UInt64], shape: Annotated[Any, TV_XlaRngBitGenerator_Tshape], dtype: TV_XlaRngBitGenerator_dtype, name, ctx):
  if dtype is None:
    dtype = _dtypes.uint64
  dtype = _execute.make_type(dtype, "dtype")
  _attr_Tshape, (shape,) = _execute.args_to_matching_eager([shape], ctx, [_dtypes.int32, _dtypes.int64, ], _dtypes.int32)
  algorithm = _ops.convert_to_tensor(algorithm, _dtypes.int32)
  initial_state = _ops.convert_to_tensor(initial_state, _dtypes.uint64)
  _inputs_flat = [algorithm, initial_state, shape]
  _attrs = ("dtype", dtype, "Tshape", _attr_Tshape)
  _result = _execute.execute(b"XlaRngBitGenerator", 2, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "XlaRngBitGenerator", _inputs_flat, _attrs, _result)
  _result = _XlaRngBitGeneratorOutput._make(_result)
  return _result


TV_XlaScatter_T = TypeVar("TV_XlaScatter_T", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float32, _atypes.Float64, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.UInt16, _atypes.UInt32, _atypes.UInt64, _atypes.UInt8)
TV_XlaScatter_Tindices = TypeVar("TV_XlaScatter_Tindices", _atypes.Int32, _atypes.Int64)

@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('xla_scatter')
def xla_scatter(operand: Annotated[Any, TV_XlaScatter_T], scatter_indices: Annotated[Any, TV_XlaScatter_Tindices], updates: Annotated[Any, TV_XlaScatter_T], update_computation, dimension_numbers: str, indices_are_sorted: bool, name=None) -> Annotated[Any, TV_XlaScatter_T]:
  r"""Wraps the XLA Scatter operator documented at

    https://www.tensorflow.org/xla/operation_semantics#scatter.

  Args:
    operand: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`, `complex64`, `int64`, `qint8`, `quint8`, `qint32`, `bfloat16`, `qint16`, `quint16`, `uint16`, `complex128`, `half`, `uint32`, `uint64`, `bool`.
      Array to be scattered into.
    scatter_indices: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      Array containing the starting indices of the slices that must
      be scattered to.
    updates: A `Tensor`. Must have the same type as `operand`.
      Array containing the values that must be used for scattering.
    update_computation: A function decorated with @Defun.
      Computation to be used for combining the existing values in
      the input array and the updates during scatter.
    dimension_numbers: A `string`.
      A serialized xla::ScatterDimensionNumbers proto.
    indices_are_sorted: A `bool`.
      Boolean indicating if the indices are sorted.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `operand`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "XlaScatter", name, operand, scatter_indices, updates,
        "update_computation", update_computation, "dimension_numbers",
        dimension_numbers, "indices_are_sorted", indices_are_sorted)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_xla_scatter(
          (operand, scatter_indices, updates, update_computation,
          dimension_numbers, indices_are_sorted, name,), None)
      if _result is not NotImplemented:
        return _result
      return xla_scatter_eager_fallback(
          operand, scatter_indices, updates,
          update_computation=update_computation,
          dimension_numbers=dimension_numbers,
          indices_are_sorted=indices_are_sorted, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            xla_scatter, (), dict(operand=operand,
                                  scatter_indices=scatter_indices,
                                  updates=updates,
                                  update_computation=update_computation,
                                  dimension_numbers=dimension_numbers,
                                  indices_are_sorted=indices_are_sorted,
                                  name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_xla_scatter(
        (operand, scatter_indices, updates, update_computation,
        dimension_numbers, indices_are_sorted, name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  dimension_numbers = _execute.make_str(dimension_numbers, "dimension_numbers")
  indices_are_sorted = _execute.make_bool(indices_are_sorted, "indices_are_sorted")
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "XlaScatter", operand=operand, scatter_indices=scatter_indices,
                      updates=updates, update_computation=update_computation,
                      dimension_numbers=dimension_numbers,
                      indices_are_sorted=indices_are_sorted, name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          xla_scatter, (), dict(operand=operand,
                                scatter_indices=scatter_indices,
                                updates=updates,
                                update_computation=update_computation,
                                dimension_numbers=dimension_numbers,
                                indices_are_sorted=indices_are_sorted,
                                name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("update_computation", _op.get_attr("update_computation"),
              "dimension_numbers", _op.get_attr("dimension_numbers"),
              "indices_are_sorted", _op._get_attr_bool("indices_are_sorted"),
              "T", _op._get_attr_type("T"), "Tindices",
              _op._get_attr_type("Tindices"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "XlaScatter", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

XlaScatter = tf_export("raw_ops.XlaScatter")(_ops.to_raw_op(xla_scatter))
_dispatcher_for_xla_scatter = xla_scatter._tf_type_based_dispatcher.Dispatch


def xla_scatter_eager_fallback(operand: Annotated[Any, TV_XlaScatter_T], scatter_indices: Annotated[Any, TV_XlaScatter_Tindices], updates: Annotated[Any, TV_XlaScatter_T], update_computation, dimension_numbers: str, indices_are_sorted: bool, name, ctx) -> Annotated[Any, TV_XlaScatter_T]:
  dimension_numbers = _execute.make_str(dimension_numbers, "dimension_numbers")
  indices_are_sorted = _execute.make_bool(indices_are_sorted, "indices_are_sorted")
  _attr_T, _inputs_T = _execute.args_to_matching_eager([operand, updates], ctx, [_dtypes.float32, _dtypes.float64, _dtypes.int32, _dtypes.uint8, _dtypes.int16, _dtypes.int8, _dtypes.complex64, _dtypes.int64, _dtypes.qint8, _dtypes.quint8, _dtypes.qint32, _dtypes.bfloat16, _dtypes.qint16, _dtypes.quint16, _dtypes.uint16, _dtypes.complex128, _dtypes.half, _dtypes.uint32, _dtypes.uint64, _dtypes.bool, ])
  (operand, updates) = _inputs_T
  _attr_Tindices, (scatter_indices,) = _execute.args_to_matching_eager([scatter_indices], ctx, [_dtypes.int32, _dtypes.int64, ])
  _inputs_flat = [operand, scatter_indices, updates]
  _attrs = ("update_computation", update_computation, "dimension_numbers",
  dimension_numbers, "indices_are_sorted", indices_are_sorted, "T", _attr_T,
  "Tindices", _attr_Tindices)
  _result = _execute.execute(b"XlaScatter", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "XlaScatter", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_XlaSelectAndScatter_T = TypeVar("TV_XlaSelectAndScatter_T", _atypes.BFloat16, _atypes.Complex128, _atypes.Complex64, _atypes.Float32, _atypes.Float64, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.UInt16, _atypes.UInt32, _atypes.UInt64, _atypes.UInt8)
TV_XlaSelectAndScatter_Tindices = TypeVar("TV_XlaSelectAndScatter_Tindices", _atypes.Int32, _atypes.Int64)

@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('xla_select_and_scatter')
def xla_select_and_scatter(operand: Annotated[Any, TV_XlaSelectAndScatter_T], window_dimensions: Annotated[Any, TV_XlaSelectAndScatter_Tindices], window_strides: Annotated[Any, TV_XlaSelectAndScatter_Tindices], padding: Annotated[Any, TV_XlaSelectAndScatter_Tindices], source: Annotated[Any, TV_XlaSelectAndScatter_T], init_value: Annotated[Any, TV_XlaSelectAndScatter_T], select, scatter, name=None) -> Annotated[Any, TV_XlaSelectAndScatter_T]:
  r"""Wraps the XLA SelectAndScatter operator, documented at

   https://www.tensorflow.org/performance/xla/operation_semantics#selectandscatter
  .

  Args:
    operand: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`, `complex64`, `int64`, `qint8`, `quint8`, `qint32`, `bfloat16`, `qint16`, `quint16`, `uint16`, `complex128`, `half`, `uint32`, `uint64`.
      the input tensor
    window_dimensions: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      the shape of the window
    window_strides: A `Tensor`. Must have the same type as `window_dimensions`.
      the inter-window strides
    padding: A `Tensor`. Must have the same type as `window_dimensions`.
      the padding to apply at the start and end of each input dimensions
    source: A `Tensor`. Must have the same type as `operand`.
      a tensor of values to scatter
    init_value: A `Tensor`. Must have the same type as `operand`.
      a scalar representing the initial value for the output tensor
    select: A function decorated with @Defun. a selection function to apply
    scatter: A function decorated with @Defun. a scatter function to apply
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `operand`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "XlaSelectAndScatter", name, operand, window_dimensions,
        window_strides, padding, source, init_value, "select", select,
        "scatter", scatter)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_xla_select_and_scatter(
          (operand, window_dimensions, window_strides, padding, source,
          init_value, select, scatter, name,), None)
      if _result is not NotImplemented:
        return _result
      return xla_select_and_scatter_eager_fallback(
          operand, window_dimensions, window_strides, padding, source,
          init_value, select=select, scatter=scatter, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            xla_select_and_scatter, (), dict(operand=operand,
                                             window_dimensions=window_dimensions,
                                             window_strides=window_strides,
                                             padding=padding, source=source,
                                             init_value=init_value,
                                             select=select, scatter=scatter,
                                             name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_xla_select_and_scatter(
        (operand, window_dimensions, window_strides, padding, source,
        init_value, select, scatter, name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "XlaSelectAndScatter", operand=operand,
                               window_dimensions=window_dimensions,
                               window_strides=window_strides, padding=padding,
                               source=source, init_value=init_value,
                               select=select, scatter=scatter, name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          xla_select_and_scatter, (), dict(operand=operand,
                                           window_dimensions=window_dimensions,
                                           window_strides=window_strides,
                                           padding=padding, source=source,
                                           init_value=init_value,
                                           select=select, scatter=scatter,
                                           name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"), "Tindices",
              _op._get_attr_type("Tindices"), "select",
              _op.get_attr("select"), "scatter", _op.get_attr("scatter"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "XlaSelectAndScatter", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

XlaSelectAndScatter = tf_export("raw_ops.XlaSelectAndScatter")(_ops.to_raw_op(xla_select_and_scatter))
_dispatcher_for_xla_select_and_scatter = xla_select_and_scatter._tf_type_based_dispatcher.Dispatch


def xla_select_and_scatter_eager_fallback(operand: Annotated[Any, TV_XlaSelectAndScatter_T], window_dimensions: Annotated[Any, TV_XlaSelectAndScatter_Tindices], window_strides: Annotated[Any, TV_XlaSelectAndScatter_Tindices], padding: Annotated[Any, TV_XlaSelectAndScatter_Tindices], source: Annotated[Any, TV_XlaSelectAndScatter_T], init_value: Annotated[Any, TV_XlaSelectAndScatter_T], select, scatter, name, ctx) -> Annotated[Any, TV_XlaSelectAndScatter_T]:
  _attr_T, _inputs_T = _execute.args_to_matching_eager([operand, source, init_value], ctx, [_dtypes.float32, _dtypes.float64, _dtypes.int32, _dtypes.uint8, _dtypes.int16, _dtypes.int8, _dtypes.complex64, _dtypes.int64, _dtypes.qint8, _dtypes.quint8, _dtypes.qint32, _dtypes.bfloat16, _dtypes.qint16, _dtypes.quint16, _dtypes.uint16, _dtypes.complex128, _dtypes.half, _dtypes.uint32, _dtypes.uint64, ])
  (operand, source, init_value) = _inputs_T
  _attr_Tindices, _inputs_Tindices = _execute.args_to_matching_eager([window_dimensions, window_strides, padding], ctx, [_dtypes.int32, _dtypes.int64, ])
  (window_dimensions, window_strides, padding) = _inputs_Tindices
  _inputs_flat = [operand, window_dimensions, window_strides, padding, source, init_value]
  _attrs = ("T", _attr_T, "Tindices", _attr_Tindices, "select", select,
  "scatter", scatter)
  _result = _execute.execute(b"XlaSelectAndScatter", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "XlaSelectAndScatter", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

_XlaSelfAdjointEigOutput = collections.namedtuple(
    "XlaSelfAdjointEig",
    ["w", "v"])


TV_XlaSelfAdjointEig_T = TypeVar("TV_XlaSelfAdjointEig_T", _atypes.BFloat16, _atypes.Complex128, _atypes.Complex64, _atypes.Float32, _atypes.Float64, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.UInt16, _atypes.UInt32, _atypes.UInt64, _atypes.UInt8)

@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('xla_self_adjoint_eig')
def xla_self_adjoint_eig(a: Annotated[Any, TV_XlaSelfAdjointEig_T], lower: bool, max_iter: int, epsilon: float, name=None):
  r"""Computes the eigen decomposition of a batch of self-adjoint matrices

  (Note: Only real inputs are supported).

  Computes the eigenvalues and eigenvectors of the innermost N-by-N matrices in
  tensor such that tensor[...,:,:] * v[..., :,i] = e[..., i] * v[...,:,i], for
  i=0...N-1.

  Args:
    a: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`, `complex64`, `int64`, `qint8`, `quint8`, `qint32`, `bfloat16`, `qint16`, `quint16`, `uint16`, `complex128`, `half`, `uint32`, `uint64`.
      the input tensor.
    lower: A `bool`.
      a boolean specifies whether the calculation is done with the lower
      triangular part or the upper triangular part.
    max_iter: An `int`.
      maximum number of sweep update, i.e., the whole lower triangular
      part or upper triangular part based on parameter lower. Heuristically, it has
      been argued that approximately logN sweeps are needed in practice (Ref: Golub &
      van Loan "Matrix Computation").
    epsilon: A `float`. the tolerance ratio.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (w, v).

    w: A `Tensor`. Has the same type as `a`. The eigenvalues in ascending order, each repeated according to its
      multiplicity.
    v: A `Tensor`. Has the same type as `a`. The column v[..., :, i] is the normalized eigenvector corresponding to the
      eigenvalue w[..., i].
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "XlaSelfAdjointEig", name, a, "lower", lower, "max_iter",
        max_iter, "epsilon", epsilon)
      _result = _XlaSelfAdjointEigOutput._make(_result)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_xla_self_adjoint_eig(
          (a, lower, max_iter, epsilon, name,), None)
      if _result is not NotImplemented:
        return _result
      return xla_self_adjoint_eig_eager_fallback(
          a, lower=lower, max_iter=max_iter, epsilon=epsilon, name=name,
          ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            xla_self_adjoint_eig, (), dict(a=a, lower=lower,
                                           max_iter=max_iter, epsilon=epsilon,
                                           name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_xla_self_adjoint_eig(
        (a, lower, max_iter, epsilon, name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  lower = _execute.make_bool(lower, "lower")
  max_iter = _execute.make_int(max_iter, "max_iter")
  epsilon = _execute.make_float(epsilon, "epsilon")
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "XlaSelfAdjointEig", a=a, lower=lower, max_iter=max_iter,
                             epsilon=epsilon, name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          xla_self_adjoint_eig, (), dict(a=a, lower=lower, max_iter=max_iter,
                                         epsilon=epsilon, name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("lower", _op._get_attr_bool("lower"), "max_iter",
              _op._get_attr_int("max_iter"), "epsilon",
              _op.get_attr("epsilon"), "T", _op._get_attr_type("T"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "XlaSelfAdjointEig", _inputs_flat, _attrs, _result)
  _result = _XlaSelfAdjointEigOutput._make(_result)
  return _result

XlaSelfAdjointEig = tf_export("raw_ops.XlaSelfAdjointEig")(_ops.to_raw_op(xla_self_adjoint_eig))
_dispatcher_for_xla_self_adjoint_eig = xla_self_adjoint_eig._tf_type_based_dispatcher.Dispatch


def xla_self_adjoint_eig_eager_fallback(a: Annotated[Any, TV_XlaSelfAdjointEig_T], lower: bool, max_iter: int, epsilon: float, name, ctx):
  lower = _execute.make_bool(lower, "lower")
  max_iter = _execute.make_int(max_iter, "max_iter")
  epsilon = _execute.make_float(epsilon, "epsilon")
  _attr_T, (a,) = _execute.args_to_matching_eager([a], ctx, [_dtypes.float32, _dtypes.float64, _dtypes.int32, _dtypes.uint8, _dtypes.int16, _dtypes.int8, _dtypes.complex64, _dtypes.int64, _dtypes.qint8, _dtypes.quint8, _dtypes.qint32, _dtypes.bfloat16, _dtypes.qint16, _dtypes.quint16, _dtypes.uint16, _dtypes.complex128, _dtypes.half, _dtypes.uint32, _dtypes.uint64, ])
  _inputs_flat = [a]
  _attrs = ("lower", lower, "max_iter", max_iter, "epsilon", epsilon, "T",
  _attr_T)
  _result = _execute.execute(b"XlaSelfAdjointEig", 2, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "XlaSelfAdjointEig", _inputs_flat, _attrs, _result)
  _result = _XlaSelfAdjointEigOutput._make(_result)
  return _result


TV_XlaSend_T = TypeVar("TV_XlaSend_T", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float16, _atypes.Float32, _atypes.Float64, _atypes.Float8e4m3fn, _atypes.Float8e5m2, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int4, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.Resource, _atypes.String, _atypes.UInt16, _atypes.UInt32, _atypes.UInt4, _atypes.UInt64, _atypes.UInt8, _atypes.Variant)

@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('xla_send')
def xla_send(tensor: Annotated[Any, TV_XlaSend_T], tensor_name: str, name=None):
  r"""Sends the named tensor to another XLA computation. Wraps the XLA Send operator

  documented at
   https://www.tensorflow.org/performance/xla/operation_semantics#send .

  Args:
    tensor: A `Tensor`. The tensor to send.
    tensor_name: A `string`. A string key that identifies the channel.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "XlaSend", name, tensor, "tensor_name", tensor_name)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_xla_send(
          (tensor, tensor_name, name,), None)
      if _result is not NotImplemented:
        return _result
      return xla_send_eager_fallback(
          tensor, tensor_name=tensor_name, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            xla_send, (), dict(tensor=tensor, tensor_name=tensor_name,
                               name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_xla_send(
        (tensor, tensor_name, name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  tensor_name = _execute.make_str(tensor_name, "tensor_name")
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "XlaSend", tensor=tensor, tensor_name=tensor_name, name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          xla_send, (), dict(tensor=tensor, tensor_name=tensor_name,
                             name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  return _op
XlaSend = tf_export("raw_ops.XlaSend")(_ops.to_raw_op(xla_send))
_dispatcher_for_xla_send = xla_send._tf_type_based_dispatcher.Dispatch


def xla_send_eager_fallback(tensor: Annotated[Any, TV_XlaSend_T], tensor_name: str, name, ctx):
  tensor_name = _execute.make_str(tensor_name, "tensor_name")
  _attr_T, (tensor,) = _execute.args_to_matching_eager([tensor], ctx, [])
  _inputs_flat = [tensor]
  _attrs = ("T", _attr_T, "tensor_name", tensor_name)
  _result = _execute.execute(b"XlaSend", 0, inputs=_inputs_flat, attrs=_attrs,
                             ctx=ctx, name=name)
  _result = None
  return _result


@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('xla_set_bound')
def xla_set_bound(input: Annotated[Any, _atypes.Int32], bound: Annotated[Any, _atypes.Int32], name=None) -> Annotated[Any, _atypes.Int32]:
  r"""Set a bound for the given input value as a hint to Xla compiler,

          returns the same value.

  Args:
    input: A `Tensor` of type `int32`.
    bound: A `Tensor` of type `int32`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `int32`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "XlaSetBound", name, input, bound)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_xla_set_bound(
          (input, bound, name,), None)
      if _result is not NotImplemented:
        return _result
      return xla_set_bound_eager_fallback(
          input, bound, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            xla_set_bound, (), dict(input=input, bound=bound, name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_xla_set_bound(
        (input, bound, name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "XlaSetBound", input=input, bound=bound, name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          xla_set_bound, (), dict(input=input, bound=bound, name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ()
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "XlaSetBound", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

XlaSetBound = tf_export("raw_ops.XlaSetBound")(_ops.to_raw_op(xla_set_bound))
_dispatcher_for_xla_set_bound = xla_set_bound._tf_type_based_dispatcher.Dispatch


def xla_set_bound_eager_fallback(input: Annotated[Any, _atypes.Int32], bound: Annotated[Any, _atypes.Int32], name, ctx) -> Annotated[Any, _atypes.Int32]:
  input = _ops.convert_to_tensor(input, _dtypes.int32)
  bound = _ops.convert_to_tensor(bound, _dtypes.int32)
  _inputs_flat = [input, bound]
  _attrs = None
  _result = _execute.execute(b"XlaSetBound", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "XlaSetBound", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_XlaSetDynamicDimensionSize_T = TypeVar("TV_XlaSetDynamicDimensionSize_T", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float16, _atypes.Float32, _atypes.Float64, _atypes.Float8e4m3fn, _atypes.Float8e5m2, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int4, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.Resource, _atypes.String, _atypes.UInt16, _atypes.UInt32, _atypes.UInt4, _atypes.UInt64, _atypes.UInt8, _atypes.Variant)

@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('xla_set_dynamic_dimension_size')
def xla_set_dynamic_dimension_size(input: Annotated[Any, TV_XlaSetDynamicDimensionSize_T], dim_index: Annotated[Any, _atypes.Int32], size: Annotated[Any, _atypes.Int32], name=None) -> Annotated[Any, TV_XlaSetDynamicDimensionSize_T]:
  r"""Make a static dimension into a xla bounded dynamic dimension.

          The current static dimension size will become the bound and the second
          operand becomes the dynamic size of the dimension.

  Args:
    input: A `Tensor`.
    dim_index: A `Tensor` of type `int32`.
    size: A `Tensor` of type `int32`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "XlaSetDynamicDimensionSize", name, input, dim_index, size)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_xla_set_dynamic_dimension_size(
          (input, dim_index, size, name,), None)
      if _result is not NotImplemented:
        return _result
      return xla_set_dynamic_dimension_size_eager_fallback(
          input, dim_index, size, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            xla_set_dynamic_dimension_size, (), dict(input=input,
                                                     dim_index=dim_index,
                                                     size=size, name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_xla_set_dynamic_dimension_size(
        (input, dim_index, size, name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "XlaSetDynamicDimensionSize", input=input, dim_index=dim_index,
                                      size=size, name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          xla_set_dynamic_dimension_size, (), dict(input=input,
                                                   dim_index=dim_index,
                                                   size=size, name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "XlaSetDynamicDimensionSize", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

XlaSetDynamicDimensionSize = tf_export("raw_ops.XlaSetDynamicDimensionSize")(_ops.to_raw_op(xla_set_dynamic_dimension_size))
_dispatcher_for_xla_set_dynamic_dimension_size = xla_set_dynamic_dimension_size._tf_type_based_dispatcher.Dispatch


def xla_set_dynamic_dimension_size_eager_fallback(input: Annotated[Any, TV_XlaSetDynamicDimensionSize_T], dim_index: Annotated[Any, _atypes.Int32], size: Annotated[Any, _atypes.Int32], name, ctx) -> Annotated[Any, TV_XlaSetDynamicDimensionSize_T]:
  _attr_T, (input,) = _execute.args_to_matching_eager([input], ctx, [])
  dim_index = _ops.convert_to_tensor(dim_index, _dtypes.int32)
  size = _ops.convert_to_tensor(size, _dtypes.int32)
  _inputs_flat = [input, dim_index, size]
  _attrs = ("T", _attr_T)
  _result = _execute.execute(b"XlaSetDynamicDimensionSize", 1,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "XlaSetDynamicDimensionSize", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_XlaSharding_T = TypeVar("TV_XlaSharding_T", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float16, _atypes.Float32, _atypes.Float64, _atypes.Float8e4m3fn, _atypes.Float8e5m2, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int4, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.Resource, _atypes.String, _atypes.UInt16, _atypes.UInt32, _atypes.UInt4, _atypes.UInt64, _atypes.UInt8, _atypes.Variant)

@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('xla_sharding')
def xla_sharding(input: Annotated[Any, TV_XlaSharding_T], sharding:str="", unspecified_dims=[], name=None) -> Annotated[Any, TV_XlaSharding_T]:
  r"""An op which shards the input based on the given sharding attribute. It can

  selectively annotate a subset of tensor dimensions by skipping unspecified_dims,
  and the sharding annotation should be replicated in those dims.

  Args:
    input: A `Tensor`.
    sharding: An optional `string`. Defaults to `""`.
    unspecified_dims: An optional list of `ints`. Defaults to `[]`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "XlaSharding", name, input, "sharding", sharding,
        "unspecified_dims", unspecified_dims)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_xla_sharding(
          (input, sharding, unspecified_dims, name,), None)
      if _result is not NotImplemented:
        return _result
      return xla_sharding_eager_fallback(
          input, sharding=sharding, unspecified_dims=unspecified_dims,
          name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            xla_sharding, (), dict(input=input, sharding=sharding,
                                   unspecified_dims=unspecified_dims,
                                   name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_xla_sharding(
        (input, sharding, unspecified_dims, name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  if sharding is None:
    sharding = ""
  sharding = _execute.make_str(sharding, "sharding")
  if unspecified_dims is None:
    unspecified_dims = []
  if not isinstance(unspecified_dims, (list, tuple)):
    raise TypeError(
        "Expected list for 'unspecified_dims' argument to "
        "'xla_sharding' Op, not %r." % unspecified_dims)
  unspecified_dims = [_execute.make_int(_i, "unspecified_dims") for _i in unspecified_dims]
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "XlaSharding", input=input, sharding=sharding,
                       unspecified_dims=unspecified_dims, name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          xla_sharding, (), dict(input=input, sharding=sharding,
                                 unspecified_dims=unspecified_dims, name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"), "sharding",
              _op.get_attr("sharding"), "unspecified_dims",
              _op.get_attr("unspecified_dims"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "XlaSharding", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

XlaSharding = tf_export("raw_ops.XlaSharding")(_ops.to_raw_op(xla_sharding))
_dispatcher_for_xla_sharding = xla_sharding._tf_type_based_dispatcher.Dispatch


def xla_sharding_eager_fallback(input: Annotated[Any, TV_XlaSharding_T], sharding: str, unspecified_dims, name, ctx) -> Annotated[Any, TV_XlaSharding_T]:
  if sharding is None:
    sharding = ""
  sharding = _execute.make_str(sharding, "sharding")
  if unspecified_dims is None:
    unspecified_dims = []
  if not isinstance(unspecified_dims, (list, tuple)):
    raise TypeError(
        "Expected list for 'unspecified_dims' argument to "
        "'xla_sharding' Op, not %r." % unspecified_dims)
  unspecified_dims = [_execute.make_int(_i, "unspecified_dims") for _i in unspecified_dims]
  _attr_T, (input,) = _execute.args_to_matching_eager([input], ctx, [])
  _inputs_flat = [input]
  _attrs = ("T", _attr_T, "sharding", sharding, "unspecified_dims",
  unspecified_dims)
  _result = _execute.execute(b"XlaSharding", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "XlaSharding", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_XlaSort_T = TypeVar("TV_XlaSort_T", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float16, _atypes.Float32, _atypes.Float64, _atypes.Float8e4m3fn, _atypes.Float8e5m2, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int4, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.Resource, _atypes.String, _atypes.UInt16, _atypes.UInt32, _atypes.UInt4, _atypes.UInt64, _atypes.UInt8, _atypes.Variant)

@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('xla_sort')
def xla_sort(input: Annotated[Any, TV_XlaSort_T], name=None) -> Annotated[Any, TV_XlaSort_T]:
  r"""Wraps the XLA Sort operator, documented at

   https://www.tensorflow.org/performance/xla/operation_semantics#sort
  .

  Sorts a tensor. Currently only sorts in ascending order are supported.

  Args:
    input: A `Tensor`. A `Tensor` of type T.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`. A `Tensor` of type T.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "XlaSort", name, input)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_xla_sort(
          (input, name,), None)
      if _result is not NotImplemented:
        return _result
      return xla_sort_eager_fallback(
          input, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            xla_sort, (), dict(input=input, name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_xla_sort(
        (input, name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "XlaSort", input=input, name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          xla_sort, (), dict(input=input, name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "XlaSort", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

XlaSort = tf_export("raw_ops.XlaSort")(_ops.to_raw_op(xla_sort))
_dispatcher_for_xla_sort = xla_sort._tf_type_based_dispatcher.Dispatch


def xla_sort_eager_fallback(input: Annotated[Any, TV_XlaSort_T], name, ctx) -> Annotated[Any, TV_XlaSort_T]:
  _attr_T, (input,) = _execute.args_to_matching_eager([input], ctx, [])
  _inputs_flat = [input]
  _attrs = ("T", _attr_T)
  _result = _execute.execute(b"XlaSort", 1, inputs=_inputs_flat, attrs=_attrs,
                             ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "XlaSort", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_XlaSpmdFullToShardShape_T = TypeVar("TV_XlaSpmdFullToShardShape_T", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float16, _atypes.Float32, _atypes.Float64, _atypes.Float8e4m3fn, _atypes.Float8e5m2, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int4, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.Resource, _atypes.String, _atypes.UInt16, _atypes.UInt32, _atypes.UInt4, _atypes.UInt64, _atypes.UInt8, _atypes.Variant)

@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('xla_spmd_full_to_shard_shape')
def xla_spmd_full_to_shard_shape(input: Annotated[Any, TV_XlaSpmdFullToShardShape_T], manual_sharding: str, dim:int=-1, unspecified_dims=[], name=None) -> Annotated[Any, TV_XlaSpmdFullToShardShape_T]:
  r"""An op used by XLA SPMD partitioner to switch from automatic partitioning to

  manual partitioning. It annotates the input (full-shape, to be automatically
  partitioned) with the same sharding used by manual partitioning, and outputs a
  shard-shaped tensor to be consumed by later manually-partitioned ops. If the
  shape is not evenly partitionable, the padding region will be masked with 0s.
  The conversion can happen partially in subgroups, by specifying the dim
  attribute, where only that dim will be converted.

  Args:
    input: A `Tensor`.
    manual_sharding: A `string`.
    dim: An optional `int`. Defaults to `-1`.
    unspecified_dims: An optional list of `ints`. Defaults to `[]`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "XlaSpmdFullToShardShape", name, input, "manual_sharding",
        manual_sharding, "dim", dim, "unspecified_dims", unspecified_dims)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_xla_spmd_full_to_shard_shape(
          (input, manual_sharding, dim, unspecified_dims, name,), None)
      if _result is not NotImplemented:
        return _result
      return xla_spmd_full_to_shard_shape_eager_fallback(
          input, manual_sharding=manual_sharding, dim=dim,
          unspecified_dims=unspecified_dims, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            xla_spmd_full_to_shard_shape, (), dict(input=input,
                                                   manual_sharding=manual_sharding,
                                                   dim=dim,
                                                   unspecified_dims=unspecified_dims,
                                                   name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_xla_spmd_full_to_shard_shape(
        (input, manual_sharding, dim, unspecified_dims, name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  manual_sharding = _execute.make_str(manual_sharding, "manual_sharding")
  if dim is None:
    dim = -1
  dim = _execute.make_int(dim, "dim")
  if unspecified_dims is None:
    unspecified_dims = []
  if not isinstance(unspecified_dims, (list, tuple)):
    raise TypeError(
        "Expected list for 'unspecified_dims' argument to "
        "'xla_spmd_full_to_shard_shape' Op, not %r." % unspecified_dims)
  unspecified_dims = [_execute.make_int(_i, "unspecified_dims") for _i in unspecified_dims]
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "XlaSpmdFullToShardShape", input=input,
                                   manual_sharding=manual_sharding, dim=dim,
                                   unspecified_dims=unspecified_dims,
                                   name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          xla_spmd_full_to_shard_shape, (), dict(input=input,
                                                 manual_sharding=manual_sharding,
                                                 dim=dim,
                                                 unspecified_dims=unspecified_dims,
                                                 name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"), "manual_sharding",
              _op.get_attr("manual_sharding"), "dim",
              _op._get_attr_int("dim"), "unspecified_dims",
              _op.get_attr("unspecified_dims"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "XlaSpmdFullToShardShape", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

XlaSpmdFullToShardShape = tf_export("raw_ops.XlaSpmdFullToShardShape")(_ops.to_raw_op(xla_spmd_full_to_shard_shape))
_dispatcher_for_xla_spmd_full_to_shard_shape = xla_spmd_full_to_shard_shape._tf_type_based_dispatcher.Dispatch


def xla_spmd_full_to_shard_shape_eager_fallback(input: Annotated[Any, TV_XlaSpmdFullToShardShape_T], manual_sharding: str, dim: int, unspecified_dims, name, ctx) -> Annotated[Any, TV_XlaSpmdFullToShardShape_T]:
  manual_sharding = _execute.make_str(manual_sharding, "manual_sharding")
  if dim is None:
    dim = -1
  dim = _execute.make_int(dim, "dim")
  if unspecified_dims is None:
    unspecified_dims = []
  if not isinstance(unspecified_dims, (list, tuple)):
    raise TypeError(
        "Expected list for 'unspecified_dims' argument to "
        "'xla_spmd_full_to_shard_shape' Op, not %r." % unspecified_dims)
  unspecified_dims = [_execute.make_int(_i, "unspecified_dims") for _i in unspecified_dims]
  _attr_T, (input,) = _execute.args_to_matching_eager([input], ctx, [])
  _inputs_flat = [input]
  _attrs = ("T", _attr_T, "manual_sharding", manual_sharding, "dim", dim,
  "unspecified_dims", unspecified_dims)
  _result = _execute.execute(b"XlaSpmdFullToShardShape", 1,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "XlaSpmdFullToShardShape", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_XlaSpmdShardToFullShape_T = TypeVar("TV_XlaSpmdShardToFullShape_T", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float16, _atypes.Float32, _atypes.Float64, _atypes.Float8e4m3fn, _atypes.Float8e5m2, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int4, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.Resource, _atypes.String, _atypes.UInt16, _atypes.UInt32, _atypes.UInt4, _atypes.UInt64, _atypes.UInt8, _atypes.Variant)

@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('xla_spmd_shard_to_full_shape')
def xla_spmd_shard_to_full_shape(input: Annotated[Any, TV_XlaSpmdShardToFullShape_T], manual_sharding: str, full_shape, dim:int=-1, unspecified_dims=[], name=None) -> Annotated[Any, TV_XlaSpmdShardToFullShape_T]:
  r"""An op used by XLA SPMD partitioner to switch from manual partitioning to

  automatic partitioning. It converts the shard-shaped, manually partitioned input
  into full-shaped tensor to be partitioned automatically with the same sharding
  used by manual partitioning. The conversion can happen partially in subgroups,
  by specifying the dim attribute, where only that dim will be converted.

  Args:
    input: A `Tensor`.
    manual_sharding: A `string`.
    full_shape: A `tf.TensorShape` or list of `ints`.
    dim: An optional `int`. Defaults to `-1`.
    unspecified_dims: An optional list of `ints`. Defaults to `[]`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "XlaSpmdShardToFullShape", name, input, "manual_sharding",
        manual_sharding, "full_shape", full_shape, "dim", dim,
        "unspecified_dims", unspecified_dims)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_xla_spmd_shard_to_full_shape(
          (input, manual_sharding, full_shape, dim, unspecified_dims, name,),
          None)
      if _result is not NotImplemented:
        return _result
      return xla_spmd_shard_to_full_shape_eager_fallback(
          input, manual_sharding=manual_sharding, full_shape=full_shape,
          dim=dim, unspecified_dims=unspecified_dims, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            xla_spmd_shard_to_full_shape, (), dict(input=input,
                                                   manual_sharding=manual_sharding,
                                                   full_shape=full_shape,
                                                   dim=dim,
                                                   unspecified_dims=unspecified_dims,
                                                   name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_xla_spmd_shard_to_full_shape(
        (input, manual_sharding, full_shape, dim, unspecified_dims, name,),
        None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  manual_sharding = _execute.make_str(manual_sharding, "manual_sharding")
  full_shape = _execute.make_shape(full_shape, "full_shape")
  if dim is None:
    dim = -1
  dim = _execute.make_int(dim, "dim")
  if unspecified_dims is None:
    unspecified_dims = []
  if not isinstance(unspecified_dims, (list, tuple)):
    raise TypeError(
        "Expected list for 'unspecified_dims' argument to "
        "'xla_spmd_shard_to_full_shape' Op, not %r." % unspecified_dims)
  unspecified_dims = [_execute.make_int(_i, "unspecified_dims") for _i in unspecified_dims]
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "XlaSpmdShardToFullShape", input=input,
                                   manual_sharding=manual_sharding,
                                   full_shape=full_shape, dim=dim,
                                   unspecified_dims=unspecified_dims,
                                   name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          xla_spmd_shard_to_full_shape, (), dict(input=input,
                                                 manual_sharding=manual_sharding,
                                                 full_shape=full_shape,
                                                 dim=dim,
                                                 unspecified_dims=unspecified_dims,
                                                 name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"), "manual_sharding",
              _op.get_attr("manual_sharding"), "full_shape",
              _op.get_attr("full_shape"), "dim", _op._get_attr_int("dim"),
              "unspecified_dims", _op.get_attr("unspecified_dims"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "XlaSpmdShardToFullShape", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

XlaSpmdShardToFullShape = tf_export("raw_ops.XlaSpmdShardToFullShape")(_ops.to_raw_op(xla_spmd_shard_to_full_shape))
_dispatcher_for_xla_spmd_shard_to_full_shape = xla_spmd_shard_to_full_shape._tf_type_based_dispatcher.Dispatch


def xla_spmd_shard_to_full_shape_eager_fallback(input: Annotated[Any, TV_XlaSpmdShardToFullShape_T], manual_sharding: str, full_shape, dim: int, unspecified_dims, name, ctx) -> Annotated[Any, TV_XlaSpmdShardToFullShape_T]:
  manual_sharding = _execute.make_str(manual_sharding, "manual_sharding")
  full_shape = _execute.make_shape(full_shape, "full_shape")
  if dim is None:
    dim = -1
  dim = _execute.make_int(dim, "dim")
  if unspecified_dims is None:
    unspecified_dims = []
  if not isinstance(unspecified_dims, (list, tuple)):
    raise TypeError(
        "Expected list for 'unspecified_dims' argument to "
        "'xla_spmd_shard_to_full_shape' Op, not %r." % unspecified_dims)
  unspecified_dims = [_execute.make_int(_i, "unspecified_dims") for _i in unspecified_dims]
  _attr_T, (input,) = _execute.args_to_matching_eager([input], ctx, [])
  _inputs_flat = [input]
  _attrs = ("T", _attr_T, "manual_sharding", manual_sharding, "full_shape",
  full_shape, "dim", dim, "unspecified_dims", unspecified_dims)
  _result = _execute.execute(b"XlaSpmdShardToFullShape", 1,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "XlaSpmdShardToFullShape", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

_XlaSvdOutput = collections.namedtuple(
    "XlaSvd",
    ["s", "u", "v"])


TV_XlaSvd_T = TypeVar("TV_XlaSvd_T", _atypes.BFloat16, _atypes.Complex128, _atypes.Complex64, _atypes.Float32, _atypes.Float64, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.UInt16, _atypes.UInt32, _atypes.UInt64, _atypes.UInt8)

@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('xla_svd')
def xla_svd(a: Annotated[Any, TV_XlaSvd_T], max_iter: int, epsilon: float, precision_config: str, name=None):
  r"""Computes the eigen decomposition of a batch of self-adjoint matrices

  (Note: Only real inputs are supported).

  Computes the eigenvalues and eigenvectors of the innermost M-by-N matrices in
  tensor such that tensor[...,:,:] = u[..., :, :] * Diag(s[..., :]) * Transpose(v[...,:,:]).

  Args:
    a: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`, `complex64`, `int64`, `qint8`, `quint8`, `qint32`, `bfloat16`, `qint16`, `quint16`, `uint16`, `complex128`, `half`, `uint32`, `uint64`.
      the input tensor.
    max_iter: An `int`.
      maximum number of sweep update, i.e., the whole lower triangular
      part or upper triangular part based on parameter lower. Heuristically, it has
      been argued that approximately log(min (M, N)) sweeps are needed in practice
      (Ref: Golub & van Loan "Matrix Computation").
    epsilon: A `float`. the tolerance ratio.
    precision_config: A `string`. a serialized xla::PrecisionConfig proto.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (s, u, v).

    s: A `Tensor`. Has the same type as `a`. Singular values. The values are sorted in reverse order of magnitude, so
      s[..., 0] is the largest value, s[..., 1] is the second largest, etc.
    u: A `Tensor`. Has the same type as `a`. Left singular vectors.
    v: A `Tensor`. Has the same type as `a`. Right singular vectors.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "XlaSvd", name, a, "max_iter", max_iter, "epsilon", epsilon,
        "precision_config", precision_config)
      _result = _XlaSvdOutput._make(_result)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_xla_svd(
          (a, max_iter, epsilon, precision_config, name,), None)
      if _result is not NotImplemented:
        return _result
      return xla_svd_eager_fallback(
          a, max_iter=max_iter, epsilon=epsilon,
          precision_config=precision_config, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            xla_svd, (), dict(a=a, max_iter=max_iter, epsilon=epsilon,
                              precision_config=precision_config, name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_xla_svd(
        (a, max_iter, epsilon, precision_config, name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  max_iter = _execute.make_int(max_iter, "max_iter")
  epsilon = _execute.make_float(epsilon, "epsilon")
  precision_config = _execute.make_str(precision_config, "precision_config")
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "XlaSvd", a=a, max_iter=max_iter, epsilon=epsilon,
                  precision_config=precision_config, name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          xla_svd, (), dict(a=a, max_iter=max_iter, epsilon=epsilon,
                            precision_config=precision_config, name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("max_iter", _op._get_attr_int("max_iter"), "epsilon",
              _op.get_attr("epsilon"), "precision_config",
              _op.get_attr("precision_config"), "T", _op._get_attr_type("T"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "XlaSvd", _inputs_flat, _attrs, _result)
  _result = _XlaSvdOutput._make(_result)
  return _result

XlaSvd = tf_export("raw_ops.XlaSvd")(_ops.to_raw_op(xla_svd))
_dispatcher_for_xla_svd = xla_svd._tf_type_based_dispatcher.Dispatch


def xla_svd_eager_fallback(a: Annotated[Any, TV_XlaSvd_T], max_iter: int, epsilon: float, precision_config: str, name, ctx):
  max_iter = _execute.make_int(max_iter, "max_iter")
  epsilon = _execute.make_float(epsilon, "epsilon")
  precision_config = _execute.make_str(precision_config, "precision_config")
  _attr_T, (a,) = _execute.args_to_matching_eager([a], ctx, [_dtypes.float32, _dtypes.float64, _dtypes.int32, _dtypes.uint8, _dtypes.int16, _dtypes.int8, _dtypes.complex64, _dtypes.int64, _dtypes.qint8, _dtypes.quint8, _dtypes.qint32, _dtypes.bfloat16, _dtypes.qint16, _dtypes.quint16, _dtypes.uint16, _dtypes.complex128, _dtypes.half, _dtypes.uint32, _dtypes.uint64, ])
  _inputs_flat = [a]
  _attrs = ("max_iter", max_iter, "epsilon", epsilon, "precision_config",
  precision_config, "T", _attr_T)
  _result = _execute.execute(b"XlaSvd", 3, inputs=_inputs_flat, attrs=_attrs,
                             ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "XlaSvd", _inputs_flat, _attrs, _result)
  _result = _XlaSvdOutput._make(_result)
  return _result


TV_XlaVariadicReduce_T = TypeVar("TV_XlaVariadicReduce_T", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float32, _atypes.Float64, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.UInt16, _atypes.UInt32, _atypes.UInt64, _atypes.UInt8)

@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('xla_variadic_reduce')
def xla_variadic_reduce(input: Annotated[List[Any], TV_XlaVariadicReduce_T], init_value: Annotated[List[Any], TV_XlaVariadicReduce_T], dimensions_to_reduce, reducer, name=None):
  r"""Wraps the variadic XLA Reduce operator.

  Semantics are documented at
   https://www.tensorflow.org/performance/xla/operation_semantics#variadic_reduce.

  This version is limited to operands of the same dtype.
  XlaVariadicReduceV2 is a version that supports heterogeneous operands.

  Args:
    input: A list of at least 1 `Tensor` objects with the same type in: `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`, `complex64`, `int64`, `qint8`, `quint8`, `qint32`, `bfloat16`, `qint16`, `quint16`, `uint16`, `complex128`, `half`, `uint32`, `uint64`, `bool`.
      the input tensor(s)
    init_value: A list with the same length as `input` of `Tensor` objects with the same type as `input`.
      scalar initial value(s) for the reduction
    dimensions_to_reduce: A list of `ints`.
      dimension numbers over which to reduce
    reducer: A function decorated with @Defun. a reducer function to apply
    name: A name for the operation (optional).

  Returns:
    A list with the same length as `input` of `Tensor` objects with the same type as `input`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "XlaVariadicReduce", name, input, init_value,
        "dimensions_to_reduce", dimensions_to_reduce, "reducer", reducer)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_xla_variadic_reduce(
          (input, init_value, dimensions_to_reduce, reducer, name,), None)
      if _result is not NotImplemented:
        return _result
      return xla_variadic_reduce_eager_fallback(
          input, init_value, dimensions_to_reduce=dimensions_to_reduce,
          reducer=reducer, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            xla_variadic_reduce, (), dict(input=input, init_value=init_value,
                                          dimensions_to_reduce=dimensions_to_reduce,
                                          reducer=reducer, name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_xla_variadic_reduce(
        (input, init_value, dimensions_to_reduce, reducer, name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  if not isinstance(input, (list, tuple)):
    raise TypeError(
        "Expected list for 'input' argument to "
        "'xla_variadic_reduce' Op, not %r." % input)
  _attr_N = len(input)
  if not isinstance(init_value, (list, tuple)):
    raise TypeError(
        "Expected list for 'init_value' argument to "
        "'xla_variadic_reduce' Op, not %r." % init_value)
  if len(init_value) != _attr_N:
    raise ValueError(
        "List argument 'init_value' to 'xla_variadic_reduce' Op with length %d "
        "must match length %d of argument 'input'." %
        (len(init_value), _attr_N))
  if not isinstance(dimensions_to_reduce, (list, tuple)):
    raise TypeError(
        "Expected list for 'dimensions_to_reduce' argument to "
        "'xla_variadic_reduce' Op, not %r." % dimensions_to_reduce)
  dimensions_to_reduce = [_execute.make_int(_i, "dimensions_to_reduce") for _i in dimensions_to_reduce]
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "XlaVariadicReduce", input=input, init_value=init_value,
                             dimensions_to_reduce=dimensions_to_reduce,
                             reducer=reducer, name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          xla_variadic_reduce, (), dict(input=input, init_value=init_value,
                                        dimensions_to_reduce=dimensions_to_reduce,
                                        reducer=reducer, name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("N", _op._get_attr_int("N"), "T", _op._get_attr_type("T"),
              "dimensions_to_reduce", _op.get_attr("dimensions_to_reduce"),
              "reducer", _op.get_attr("reducer"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "XlaVariadicReduce", _inputs_flat, _attrs, _result)
  return _result

XlaVariadicReduce = tf_export("raw_ops.XlaVariadicReduce")(_ops.to_raw_op(xla_variadic_reduce))
_dispatcher_for_xla_variadic_reduce = xla_variadic_reduce._tf_type_based_dispatcher.Dispatch


def xla_variadic_reduce_eager_fallback(input: Annotated[List[Any], TV_XlaVariadicReduce_T], init_value: Annotated[List[Any], TV_XlaVariadicReduce_T], dimensions_to_reduce, reducer, name, ctx):
  if not isinstance(input, (list, tuple)):
    raise TypeError(
        "Expected list for 'input' argument to "
        "'xla_variadic_reduce' Op, not %r." % input)
  _attr_N = len(input)
  if not isinstance(init_value, (list, tuple)):
    raise TypeError(
        "Expected list for 'init_value' argument to "
        "'xla_variadic_reduce' Op, not %r." % init_value)
  if len(init_value) != _attr_N:
    raise ValueError(
        "List argument 'init_value' to 'xla_variadic_reduce' Op with length %d "
        "must match length %d of argument 'input'." %
        (len(init_value), _attr_N))
  if not isinstance(dimensions_to_reduce, (list, tuple)):
    raise TypeError(
        "Expected list for 'dimensions_to_reduce' argument to "
        "'xla_variadic_reduce' Op, not %r." % dimensions_to_reduce)
  dimensions_to_reduce = [_execute.make_int(_i, "dimensions_to_reduce") for _i in dimensions_to_reduce]
  _attr_T, _inputs_T = _execute.args_to_matching_eager(list(input) + list(init_value), ctx, [_dtypes.float32, _dtypes.float64, _dtypes.int32, _dtypes.uint8, _dtypes.int16, _dtypes.int8, _dtypes.complex64, _dtypes.int64, _dtypes.qint8, _dtypes.quint8, _dtypes.qint32, _dtypes.bfloat16, _dtypes.qint16, _dtypes.quint16, _dtypes.uint16, _dtypes.complex128, _dtypes.half, _dtypes.uint32, _dtypes.uint64, _dtypes.bool, ])
  _inputs_T = [_inputs_T[:_attr_N]] + _inputs_T[_attr_N:]
  _inputs_T = _inputs_T[:1] + [_inputs_T[1:]]
  (input, init_value) = _inputs_T
  _inputs_flat = list(input) + list(init_value)
  _attrs = ("N", _attr_N, "T", _attr_T, "dimensions_to_reduce",
  dimensions_to_reduce, "reducer", reducer)
  _result = _execute.execute(b"XlaVariadicReduce", _attr_N,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "XlaVariadicReduce", _inputs_flat, _attrs, _result)
  return _result


@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('xla_variadic_reduce_v2')
def xla_variadic_reduce_v2(inputs, init_values, dimensions_to_reduce, reducer, name=None):
  r"""Wraps the variadic XLA Reduce operator.

  Semantics are documented at
   https://www.tensorflow.org/performance/xla/operation_semantics#variadic_reduce.

  This is an expanded version of XlaVariadicReduce, with support for
  operands of different dtypes, and improved shape inference.

  Args:
    inputs: A list of `Tensor` objects. the input tensor(s)
    init_values: A list of `Tensor` objects. Must have the same type as `inputs`.
      scalar initial value(s) for the reduction
    dimensions_to_reduce: A list of `ints`.
      dimension numbers over which to reduce
    reducer: A function decorated with @Defun. a reducer function to apply
    name: A name for the operation (optional).

  Returns:
    A list of `Tensor` objects. Has the same type as `inputs`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "XlaVariadicReduceV2", name, inputs, init_values,
        "dimensions_to_reduce", dimensions_to_reduce, "reducer", reducer)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_xla_variadic_reduce_v2(
          (inputs, init_values, dimensions_to_reduce, reducer, name,), None)
      if _result is not NotImplemented:
        return _result
      return xla_variadic_reduce_v2_eager_fallback(
          inputs, init_values, dimensions_to_reduce=dimensions_to_reduce,
          reducer=reducer, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            xla_variadic_reduce_v2, (), dict(inputs=inputs,
                                             init_values=init_values,
                                             dimensions_to_reduce=dimensions_to_reduce,
                                             reducer=reducer, name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_xla_variadic_reduce_v2(
        (inputs, init_values, dimensions_to_reduce, reducer, name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  if not isinstance(dimensions_to_reduce, (list, tuple)):
    raise TypeError(
        "Expected list for 'dimensions_to_reduce' argument to "
        "'xla_variadic_reduce_v2' Op, not %r." % dimensions_to_reduce)
  dimensions_to_reduce = [_execute.make_int(_i, "dimensions_to_reduce") for _i in dimensions_to_reduce]
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "XlaVariadicReduceV2", inputs=inputs, init_values=init_values,
                               dimensions_to_reduce=dimensions_to_reduce,
                               reducer=reducer, name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          xla_variadic_reduce_v2, (), dict(inputs=inputs,
                                           init_values=init_values,
                                           dimensions_to_reduce=dimensions_to_reduce,
                                           reducer=reducer, name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op.get_attr("T"), "dimensions_to_reduce",
              _op.get_attr("dimensions_to_reduce"), "reducer",
              _op.get_attr("reducer"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "XlaVariadicReduceV2", _inputs_flat, _attrs, _result)
  return _result

XlaVariadicReduceV2 = tf_export("raw_ops.XlaVariadicReduceV2")(_ops.to_raw_op(xla_variadic_reduce_v2))
_dispatcher_for_xla_variadic_reduce_v2 = xla_variadic_reduce_v2._tf_type_based_dispatcher.Dispatch


def xla_variadic_reduce_v2_eager_fallback(inputs, init_values, dimensions_to_reduce, reducer, name, ctx):
  if not isinstance(dimensions_to_reduce, (list, tuple)):
    raise TypeError(
        "Expected list for 'dimensions_to_reduce' argument to "
        "'xla_variadic_reduce_v2' Op, not %r." % dimensions_to_reduce)
  dimensions_to_reduce = [_execute.make_int(_i, "dimensions_to_reduce") for _i in dimensions_to_reduce]
  _attr_T, (inputs, init_values) = _execute.args_to_mixed_eager_tensors((inputs, init_values), ctx)
  _inputs_flat = list(inputs) + list(init_values)
  _attrs = ("T", _attr_T, "dimensions_to_reduce", dimensions_to_reduce,
  "reducer", reducer)
  _result = _execute.execute(b"XlaVariadicReduceV2", len(inputs),
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "XlaVariadicReduceV2", _inputs_flat, _attrs, _result)
  return _result


@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('xla_variadic_sort')
def xla_variadic_sort(inputs, dimension: Annotated[Any, _atypes.Int32], comparator, is_stable: bool, name=None):
  r"""Wraps the XLA Sort operator, documented at

   https://www.tensorflow.org/performance/xla/operation_semantics#sort
  .

  Sorts one or more tensors, with support for custom comparator, dimension, and
  is_stable attributes.

  Args:
    inputs: A list of `Tensor` objects.
      A list of `Tensor` of identical shape but possibly different types.
    dimension: A `Tensor` of type `int32`.
      The dimension along which to sort. Must be a compile-time constant.
    comparator: A function decorated with @Defun.
      A comparator function to apply to 2*N scalars and returning a
      boolean. N is the number of sort inputs. If you want to sort in ascending
      order then the comparator should perform a less-than comparison.
    is_stable: A `bool`. Whether to use stable sort.
    name: A name for the operation (optional).

  Returns:
    A list of `Tensor` objects. Has the same type as `inputs`.
    A list of `Tensor` of same shape and types as the `input`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "XlaVariadicSort", name, inputs, dimension, "comparator",
        comparator, "is_stable", is_stable)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_xla_variadic_sort(
          (inputs, dimension, comparator, is_stable, name,), None)
      if _result is not NotImplemented:
        return _result
      return xla_variadic_sort_eager_fallback(
          inputs, dimension, comparator=comparator, is_stable=is_stable,
          name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            xla_variadic_sort, (), dict(inputs=inputs, dimension=dimension,
                                        comparator=comparator,
                                        is_stable=is_stable, name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_xla_variadic_sort(
        (inputs, dimension, comparator, is_stable, name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  is_stable = _execute.make_bool(is_stable, "is_stable")
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "XlaVariadicSort", inputs=inputs, dimension=dimension,
                           comparator=comparator, is_stable=is_stable,
                           name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          xla_variadic_sort, (), dict(inputs=inputs, dimension=dimension,
                                      comparator=comparator,
                                      is_stable=is_stable, name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op.get_attr("T"), "comparator",
              _op.get_attr("comparator"), "is_stable",
              _op._get_attr_bool("is_stable"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "XlaVariadicSort", _inputs_flat, _attrs, _result)
  return _result

XlaVariadicSort = tf_export("raw_ops.XlaVariadicSort")(_ops.to_raw_op(xla_variadic_sort))
_dispatcher_for_xla_variadic_sort = xla_variadic_sort._tf_type_based_dispatcher.Dispatch


def xla_variadic_sort_eager_fallback(inputs, dimension: Annotated[Any, _atypes.Int32], comparator, is_stable: bool, name, ctx):
  is_stable = _execute.make_bool(is_stable, "is_stable")
  _attr_T, inputs = _execute.convert_to_mixed_eager_tensors(inputs, ctx)
  dimension = _ops.convert_to_tensor(dimension, _dtypes.int32)
  _inputs_flat = list(inputs) + [dimension]
  _attrs = ("T", _attr_T, "comparator", comparator, "is_stable", is_stable)
  _result = _execute.execute(b"XlaVariadicSort", len(inputs),
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "XlaVariadicSort", _inputs_flat, _attrs, _result)
  return _result


@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('xla_while')
def xla_while(input, cond, body, name=None):
  r"""output = input; While (Cond(output)) { output = Body(output) }

  Args:
    input: A list of `Tensor` objects.
      A list of input tensors whose types are T.
    cond: A function decorated with @Defun.
      A function takes 'input' and returns a tensor.  If the tensor is
      a scalar of non-boolean, the scalar is converted to a boolean
      according to the following rule: if the scalar is a numerical
      value, non-zero means True and zero means False; if the scalar is
      a string, non-empty means True and empty means False. If the
      tensor is not a scalar, non-emptiness means True and False
      otherwise.
    body: A function decorated with @Defun.
      A function that takes a list of tensors and returns another
      list of tensors. Both lists have the same types as specified by T.
    name: A name for the operation (optional).

  Returns:
    A list of `Tensor` objects. Has the same type as `input`.
    A list of output tensors whose types are T.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "XlaWhile", name, input, "cond", cond, "body", body)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_xla_while(
          (input, cond, body, name,), None)
      if _result is not NotImplemented:
        return _result
      return xla_while_eager_fallback(
          input, cond=cond, body=body, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            xla_while, (), dict(input=input, cond=cond, body=body, name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_xla_while(
        (input, cond, body, name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "XlaWhile", input=input, cond=cond, body=body, name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          xla_while, (), dict(input=input, cond=cond, body=body, name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  _result = _outputs[:]
  if not _result:
    return _op
  if _execute.must_record_gradient():
    _attrs = ("T", _op.get_attr("T"), "cond", _op.get_attr("cond"), "body",
              _op.get_attr("body"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "XlaWhile", _inputs_flat, _attrs, _result)
  return _result

XlaWhile = tf_export("raw_ops.XlaWhile")(_ops.to_raw_op(xla_while))
_dispatcher_for_xla_while = xla_while._tf_type_based_dispatcher.Dispatch


def xla_while_eager_fallback(input, cond, body, name, ctx):
  _attr_T, input = _execute.convert_to_mixed_eager_tensors(input, ctx)
  _inputs_flat = list(input)
  _attrs = ("T", _attr_T, "cond", cond, "body", body)
  _result = _execute.execute(b"XlaWhile", len(input), inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "XlaWhile", _inputs_flat, _attrs, _result)
  return _result

