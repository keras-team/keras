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

TV_XlaClusterOutput_T = TypeVar("TV_XlaClusterOutput_T", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float16, _atypes.Float32, _atypes.Float64, _atypes.Float8e4m3fn, _atypes.Float8e5m2, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int4, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.Resource, _atypes.String, _atypes.UInt16, _atypes.UInt32, _atypes.UInt4, _atypes.UInt64, _atypes.UInt8, _atypes.Variant)

@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('xla_cluster_output')
def xla_cluster_output(input: Annotated[Any, TV_XlaClusterOutput_T], name=None) -> Annotated[Any, TV_XlaClusterOutput_T]:
  r"""Operator that connects the output of an XLA computation to other consumer graph nodes.

  Args:
    input: A `Tensor`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "XlaClusterOutput", name, input)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_xla_cluster_output(
          (input, name,), None)
      if _result is not NotImplemented:
        return _result
      return xla_cluster_output_eager_fallback(
          input, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            xla_cluster_output, (), dict(input=input, name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_xla_cluster_output(
        (input, name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "XlaClusterOutput", input=input, name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          xla_cluster_output, (), dict(input=input, name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "XlaClusterOutput", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

XlaClusterOutput = tf_export("raw_ops.XlaClusterOutput")(_ops.to_raw_op(xla_cluster_output))
_dispatcher_for_xla_cluster_output = xla_cluster_output._tf_type_based_dispatcher.Dispatch


def xla_cluster_output_eager_fallback(input: Annotated[Any, TV_XlaClusterOutput_T], name, ctx) -> Annotated[Any, TV_XlaClusterOutput_T]:
  _attr_T, (input,) = _execute.args_to_matching_eager([input], ctx, [])
  _inputs_flat = [input]
  _attrs = ("T", _attr_T)
  _result = _execute.execute(b"XlaClusterOutput", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "XlaClusterOutput", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('xla_launch')
def xla_launch(constants, args, resources: Annotated[List[Any], _atypes.Resource], Tresults, function, name=None):
  r"""XLA Launch Op. For use by the XLA JIT only.

  Args:
    constants: A list of `Tensor` objects.
    args: A list of `Tensor` objects.
    resources: A list of `Tensor` objects with type `resource`.
    Tresults: A list of `tf.DTypes`.
    function: A function decorated with @Defun.
    name: A name for the operation (optional).

  Returns:
    A list of `Tensor` objects of type `Tresults`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "XlaLaunch", name, constants, args, resources, "Tresults",
        Tresults, "function", function)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_xla_launch(
          (constants, args, resources, Tresults, function, name,), None)
      if _result is not NotImplemented:
        return _result
      return xla_launch_eager_fallback(
          constants, args, resources, Tresults=Tresults, function=function,
          name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            xla_launch, (), dict(constants=constants, args=args,
                                 resources=resources, Tresults=Tresults,
                                 function=function, name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_xla_launch(
        (constants, args, resources, Tresults, function, name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  if not isinstance(resources, (list, tuple)):
    raise TypeError(
        "Expected list for 'resources' argument to "
        "'xla_launch' Op, not %r." % resources)
  _attr_Nresources = len(resources)
  if not isinstance(Tresults, (list, tuple)):
    raise TypeError(
        "Expected list for 'Tresults' argument to "
        "'xla_launch' Op, not %r." % Tresults)
  Tresults = [_execute.make_type(_t, "Tresults") for _t in Tresults]
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "XlaLaunch", constants=constants, args=args, resources=resources,
                     Tresults=Tresults, function=function, name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          xla_launch, (), dict(constants=constants, args=args,
                               resources=resources, Tresults=Tresults,
                               function=function, name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  _result = _outputs[:]
  if not _result:
    return _op
  if _execute.must_record_gradient():
    _attrs = ("Tconstants", _op.get_attr("Tconstants"), "Targs",
              _op.get_attr("Targs"), "Nresources",
              _op._get_attr_int("Nresources"), "Tresults",
              _op.get_attr("Tresults"), "function", _op.get_attr("function"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "XlaLaunch", _inputs_flat, _attrs, _result)
  return _result

XlaLaunch = tf_export("raw_ops.XlaLaunch")(_ops.to_raw_op(xla_launch))
_dispatcher_for_xla_launch = xla_launch._tf_type_based_dispatcher.Dispatch


def xla_launch_eager_fallback(constants, args, resources: Annotated[List[Any], _atypes.Resource], Tresults, function, name, ctx):
  if not isinstance(resources, (list, tuple)):
    raise TypeError(
        "Expected list for 'resources' argument to "
        "'xla_launch' Op, not %r." % resources)
  _attr_Nresources = len(resources)
  if not isinstance(Tresults, (list, tuple)):
    raise TypeError(
        "Expected list for 'Tresults' argument to "
        "'xla_launch' Op, not %r." % Tresults)
  Tresults = [_execute.make_type(_t, "Tresults") for _t in Tresults]
  _attr_Tconstants, constants = _execute.convert_to_mixed_eager_tensors(constants, ctx)
  _attr_Targs, args = _execute.convert_to_mixed_eager_tensors(args, ctx)
  resources = _ops.convert_n_to_tensor(resources, _dtypes.resource)
  _inputs_flat = list(constants) + list(args) + list(resources)
  _attrs = ("Tconstants", _attr_Tconstants, "Targs", _attr_Targs,
  "Nresources", _attr_Nresources, "Tresults", Tresults, "function", function)
  _result = _execute.execute(b"XlaLaunch", len(Tresults), inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "XlaLaunch", _inputs_flat, _attrs, _result)
  return _result


@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('xla_launch_v2')
def xla_launch_v2(args, Tresults, constants, resources, function, name=None):
  r"""XLA Launch Op. For use by the XLA JIT only.

  Args:
    args: A list of `Tensor` objects.
    Tresults: A list of `tf.DTypes`.
    constants: A list of `ints`.
    resources: A list of `ints`.
    function: A function decorated with @Defun.
    name: A name for the operation (optional).

  Returns:
    A list of `Tensor` objects of type `Tresults`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "XlaLaunchV2", name, args, "Tresults", Tresults, "constants",
        constants, "resources", resources, "function", function)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_xla_launch_v2(
          (args, Tresults, constants, resources, function, name,), None)
      if _result is not NotImplemented:
        return _result
      return xla_launch_v2_eager_fallback(
          args, Tresults=Tresults, constants=constants, resources=resources,
          function=function, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            xla_launch_v2, (), dict(args=args, Tresults=Tresults,
                                    constants=constants, resources=resources,
                                    function=function, name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_xla_launch_v2(
        (args, Tresults, constants, resources, function, name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  if not isinstance(Tresults, (list, tuple)):
    raise TypeError(
        "Expected list for 'Tresults' argument to "
        "'xla_launch_v2' Op, not %r." % Tresults)
  Tresults = [_execute.make_type(_t, "Tresults") for _t in Tresults]
  if not isinstance(constants, (list, tuple)):
    raise TypeError(
        "Expected list for 'constants' argument to "
        "'xla_launch_v2' Op, not %r." % constants)
  constants = [_execute.make_int(_i, "constants") for _i in constants]
  if not isinstance(resources, (list, tuple)):
    raise TypeError(
        "Expected list for 'resources' argument to "
        "'xla_launch_v2' Op, not %r." % resources)
  resources = [_execute.make_int(_i, "resources") for _i in resources]
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "XlaLaunchV2", args=args, Tresults=Tresults, constants=constants,
                       resources=resources, function=function, name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          xla_launch_v2, (), dict(args=args, Tresults=Tresults,
                                  constants=constants, resources=resources,
                                  function=function, name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("Targs", _op.get_attr("Targs"), "Tresults",
              _op.get_attr("Tresults"), "constants",
              _op.get_attr("constants"), "resources",
              _op.get_attr("resources"), "function", _op.get_attr("function"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "XlaLaunchV2", _inputs_flat, _attrs, _result)
  return _result

XlaLaunchV2 = tf_export("raw_ops.XlaLaunchV2")(_ops.to_raw_op(xla_launch_v2))
_dispatcher_for_xla_launch_v2 = xla_launch_v2._tf_type_based_dispatcher.Dispatch


def xla_launch_v2_eager_fallback(args, Tresults, constants, resources, function, name, ctx):
  if not isinstance(Tresults, (list, tuple)):
    raise TypeError(
        "Expected list for 'Tresults' argument to "
        "'xla_launch_v2' Op, not %r." % Tresults)
  Tresults = [_execute.make_type(_t, "Tresults") for _t in Tresults]
  if not isinstance(constants, (list, tuple)):
    raise TypeError(
        "Expected list for 'constants' argument to "
        "'xla_launch_v2' Op, not %r." % constants)
  constants = [_execute.make_int(_i, "constants") for _i in constants]
  if not isinstance(resources, (list, tuple)):
    raise TypeError(
        "Expected list for 'resources' argument to "
        "'xla_launch_v2' Op, not %r." % resources)
  resources = [_execute.make_int(_i, "resources") for _i in resources]
  _attr_Targs, args = _execute.convert_to_mixed_eager_tensors(args, ctx)
  _inputs_flat = list(args)
  _attrs = ("Targs", _attr_Targs, "Tresults", Tresults, "constants",
  constants, "resources", resources, "function", function)
  _result = _execute.execute(b"XlaLaunchV2", len(Tresults),
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "XlaLaunchV2", _inputs_flat, _attrs, _result)
  return _result

