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

@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('delete_rpc_future_resource')
def delete_rpc_future_resource(handle: Annotated[Any, _atypes.Resource], deleter: Annotated[Any, _atypes.Variant], name=None):
  r"""TODO: add doc.

  Args:
    handle: A `Tensor` of type `resource`.
    deleter: A `Tensor` of type `variant`.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "DeleteRpcFutureResource", name, handle, deleter)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_delete_rpc_future_resource(
          (handle, deleter, name,), None)
      if _result is not NotImplemented:
        return _result
      return delete_rpc_future_resource_eager_fallback(
          handle, deleter, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            delete_rpc_future_resource, (), dict(handle=handle,
                                                 deleter=deleter, name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_delete_rpc_future_resource(
        (handle, deleter, name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "DeleteRpcFutureResource", handle=handle, deleter=deleter, name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          delete_rpc_future_resource, (), dict(handle=handle, deleter=deleter,
                                               name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  return _op
DeleteRpcFutureResource = tf_export("raw_ops.DeleteRpcFutureResource")(_ops.to_raw_op(delete_rpc_future_resource))
_dispatcher_for_delete_rpc_future_resource = delete_rpc_future_resource._tf_type_based_dispatcher.Dispatch


def delete_rpc_future_resource_eager_fallback(handle: Annotated[Any, _atypes.Resource], deleter: Annotated[Any, _atypes.Variant], name, ctx):
  handle = _ops.convert_to_tensor(handle, _dtypes.resource)
  deleter = _ops.convert_to_tensor(deleter, _dtypes.variant)
  _inputs_flat = [handle, deleter]
  _attrs = None
  _result = _execute.execute(b"DeleteRpcFutureResource", 0,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  _result = None
  return _result

_RpcCallOutput = collections.namedtuple(
    "RpcCall",
    ["future", "deleter"])


@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('rpc_call')
def rpc_call(client: Annotated[Any, _atypes.Resource], method_name: Annotated[Any, _atypes.String], args, timeout_in_ms: Annotated[Any, _atypes.Int64], name=None):
  r"""TODO: add doc.

  Args:
    client: A `Tensor` of type `resource`.
    method_name: A `Tensor` of type `string`.
    args: A list of `Tensor` objects.
    timeout_in_ms: A `Tensor` of type `int64`.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (future, deleter).

    future: A `Tensor` of type `resource`.
    deleter: A `Tensor` of type `variant`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "RpcCall", name, client, method_name, args, timeout_in_ms)
      _result = _RpcCallOutput._make(_result)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_rpc_call(
          (client, method_name, args, timeout_in_ms, name,), None)
      if _result is not NotImplemented:
        return _result
      return rpc_call_eager_fallback(
          client, method_name, args, timeout_in_ms, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            rpc_call, (), dict(client=client, method_name=method_name,
                               args=args, timeout_in_ms=timeout_in_ms,
                               name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_rpc_call(
        (client, method_name, args, timeout_in_ms, name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "RpcCall", client=client, method_name=method_name, args=args,
                   timeout_in_ms=timeout_in_ms, name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          rpc_call, (), dict(client=client, method_name=method_name,
                             args=args, timeout_in_ms=timeout_in_ms,
                             name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("Tin", _op.get_attr("Tin"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "RpcCall", _inputs_flat, _attrs, _result)
  _result = _RpcCallOutput._make(_result)
  return _result

RpcCall = tf_export("raw_ops.RpcCall")(_ops.to_raw_op(rpc_call))
_dispatcher_for_rpc_call = rpc_call._tf_type_based_dispatcher.Dispatch


def rpc_call_eager_fallback(client: Annotated[Any, _atypes.Resource], method_name: Annotated[Any, _atypes.String], args, timeout_in_ms: Annotated[Any, _atypes.Int64], name, ctx):
  _attr_Tin, args = _execute.convert_to_mixed_eager_tensors(args, ctx)
  client = _ops.convert_to_tensor(client, _dtypes.resource)
  method_name = _ops.convert_to_tensor(method_name, _dtypes.string)
  timeout_in_ms = _ops.convert_to_tensor(timeout_in_ms, _dtypes.int64)
  _inputs_flat = [client, method_name] + list(args) + [timeout_in_ms]
  _attrs = ("Tin", _attr_Tin)
  _result = _execute.execute(b"RpcCall", 2, inputs=_inputs_flat, attrs=_attrs,
                             ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "RpcCall", _inputs_flat, _attrs, _result)
  _result = _RpcCallOutput._make(_result)
  return _result

_RpcCheckStatusOutput = collections.namedtuple(
    "RpcCheckStatus",
    ["error_code", "error"])


@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('rpc_check_status')
def rpc_check_status(status_or: Annotated[Any, _atypes.Resource], name=None):
  r"""TODO: add doc.

  Args:
    status_or: A `Tensor` of type `resource`.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (error_code, error).

    error_code: A `Tensor` of type `int64`.
    error: A `Tensor` of type `string`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "RpcCheckStatus", name, status_or)
      _result = _RpcCheckStatusOutput._make(_result)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_rpc_check_status(
          (status_or, name,), None)
      if _result is not NotImplemented:
        return _result
      return rpc_check_status_eager_fallback(
          status_or, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            rpc_check_status, (), dict(status_or=status_or, name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_rpc_check_status(
        (status_or, name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "RpcCheckStatus", status_or=status_or, name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          rpc_check_status, (), dict(status_or=status_or, name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ()
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "RpcCheckStatus", _inputs_flat, _attrs, _result)
  _result = _RpcCheckStatusOutput._make(_result)
  return _result

RpcCheckStatus = tf_export("raw_ops.RpcCheckStatus")(_ops.to_raw_op(rpc_check_status))
_dispatcher_for_rpc_check_status = rpc_check_status._tf_type_based_dispatcher.Dispatch


def rpc_check_status_eager_fallback(status_or: Annotated[Any, _atypes.Resource], name, ctx):
  status_or = _ops.convert_to_tensor(status_or, _dtypes.resource)
  _inputs_flat = [status_or]
  _attrs = None
  _result = _execute.execute(b"RpcCheckStatus", 2, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "RpcCheckStatus", _inputs_flat, _attrs, _result)
  _result = _RpcCheckStatusOutput._make(_result)
  return _result

_RpcClientOutput = collections.namedtuple(
    "RpcClient",
    ["client", "method_specs"])


@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('rpc_client')
def rpc_client(server_address: Annotated[Any, _atypes.String], timeout_in_ms: Annotated[Any, _atypes.Int64], shared_name:str="", list_registered_methods:bool=False, name=None):
  r"""TODO: add doc.

  Args:
    server_address: A `Tensor` of type `string`.
    timeout_in_ms: A `Tensor` of type `int64`.
    shared_name: An optional `string`. Defaults to `""`.
    list_registered_methods: An optional `bool`. Defaults to `False`.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (client, method_specs).

    client: A `Tensor` of type `resource`.
    method_specs: A `Tensor` of type `string`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "RpcClient", name, server_address, timeout_in_ms, "shared_name",
        shared_name, "list_registered_methods", list_registered_methods)
      _result = _RpcClientOutput._make(_result)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_rpc_client(
          (server_address, timeout_in_ms, shared_name,
          list_registered_methods, name,), None)
      if _result is not NotImplemented:
        return _result
      return rpc_client_eager_fallback(
          server_address, timeout_in_ms, shared_name=shared_name,
          list_registered_methods=list_registered_methods, name=name,
          ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            rpc_client, (), dict(server_address=server_address,
                                 timeout_in_ms=timeout_in_ms,
                                 shared_name=shared_name,
                                 list_registered_methods=list_registered_methods,
                                 name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_rpc_client(
        (server_address, timeout_in_ms, shared_name, list_registered_methods,
        name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  if shared_name is None:
    shared_name = ""
  shared_name = _execute.make_str(shared_name, "shared_name")
  if list_registered_methods is None:
    list_registered_methods = False
  list_registered_methods = _execute.make_bool(list_registered_methods, "list_registered_methods")
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "RpcClient", server_address=server_address,
                     timeout_in_ms=timeout_in_ms, shared_name=shared_name,
                     list_registered_methods=list_registered_methods,
                     name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          rpc_client, (), dict(server_address=server_address,
                               timeout_in_ms=timeout_in_ms,
                               shared_name=shared_name,
                               list_registered_methods=list_registered_methods,
                               name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("shared_name", _op.get_attr("shared_name"),
              "list_registered_methods",
              _op._get_attr_bool("list_registered_methods"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "RpcClient", _inputs_flat, _attrs, _result)
  _result = _RpcClientOutput._make(_result)
  return _result

RpcClient = tf_export("raw_ops.RpcClient")(_ops.to_raw_op(rpc_client))
_dispatcher_for_rpc_client = rpc_client._tf_type_based_dispatcher.Dispatch


def rpc_client_eager_fallback(server_address: Annotated[Any, _atypes.String], timeout_in_ms: Annotated[Any, _atypes.Int64], shared_name: str, list_registered_methods: bool, name, ctx):
  if shared_name is None:
    shared_name = ""
  shared_name = _execute.make_str(shared_name, "shared_name")
  if list_registered_methods is None:
    list_registered_methods = False
  list_registered_methods = _execute.make_bool(list_registered_methods, "list_registered_methods")
  server_address = _ops.convert_to_tensor(server_address, _dtypes.string)
  timeout_in_ms = _ops.convert_to_tensor(timeout_in_ms, _dtypes.int64)
  _inputs_flat = [server_address, timeout_in_ms]
  _attrs = ("shared_name", shared_name, "list_registered_methods",
  list_registered_methods)
  _result = _execute.execute(b"RpcClient", 2, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "RpcClient", _inputs_flat, _attrs, _result)
  _result = _RpcClientOutput._make(_result)
  return _result


@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('rpc_get_value')
def rpc_get_value(status_or: Annotated[Any, _atypes.Resource], Tout, name=None):
  r"""TODO: add doc.

  Args:
    status_or: A `Tensor` of type `resource`.
    Tout: A list of `tf.DTypes`.
    name: A name for the operation (optional).

  Returns:
    A list of `Tensor` objects of type `Tout`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "RpcGetValue", name, status_or, "Tout", Tout)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_rpc_get_value(
          (status_or, Tout, name,), None)
      if _result is not NotImplemented:
        return _result
      return rpc_get_value_eager_fallback(
          status_or, Tout=Tout, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            rpc_get_value, (), dict(status_or=status_or, Tout=Tout, name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_rpc_get_value(
        (status_or, Tout, name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  if not isinstance(Tout, (list, tuple)):
    raise TypeError(
        "Expected list for 'Tout' argument to "
        "'rpc_get_value' Op, not %r." % Tout)
  Tout = [_execute.make_type(_t, "Tout") for _t in Tout]
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "RpcGetValue", status_or=status_or, Tout=Tout, name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          rpc_get_value, (), dict(status_or=status_or, Tout=Tout, name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  _result = _outputs[:]
  if not _result:
    return _op
  if _execute.must_record_gradient():
    _attrs = ("Tout", _op.get_attr("Tout"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "RpcGetValue", _inputs_flat, _attrs, _result)
  return _result

RpcGetValue = tf_export("raw_ops.RpcGetValue")(_ops.to_raw_op(rpc_get_value))
_dispatcher_for_rpc_get_value = rpc_get_value._tf_type_based_dispatcher.Dispatch


def rpc_get_value_eager_fallback(status_or: Annotated[Any, _atypes.Resource], Tout, name, ctx):
  if not isinstance(Tout, (list, tuple)):
    raise TypeError(
        "Expected list for 'Tout' argument to "
        "'rpc_get_value' Op, not %r." % Tout)
  Tout = [_execute.make_type(_t, "Tout") for _t in Tout]
  status_or = _ops.convert_to_tensor(status_or, _dtypes.resource)
  _inputs_flat = [status_or]
  _attrs = ("Tout", Tout)
  _result = _execute.execute(b"RpcGetValue", len(Tout), inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "RpcGetValue", _inputs_flat, _attrs, _result)
  return _result


@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('rpc_server')
def rpc_server(server_address: Annotated[Any, _atypes.String], name=None) -> Annotated[Any, _atypes.Resource]:
  r"""TODO: add doc.

  Args:
    server_address: A `Tensor` of type `string`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `resource`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "RpcServer", name, server_address)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_rpc_server(
          (server_address, name,), None)
      if _result is not NotImplemented:
        return _result
      return rpc_server_eager_fallback(
          server_address, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            rpc_server, (), dict(server_address=server_address, name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_rpc_server(
        (server_address, name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "RpcServer", server_address=server_address, name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          rpc_server, (), dict(server_address=server_address, name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ()
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "RpcServer", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

RpcServer = tf_export("raw_ops.RpcServer")(_ops.to_raw_op(rpc_server))
_dispatcher_for_rpc_server = rpc_server._tf_type_based_dispatcher.Dispatch


def rpc_server_eager_fallback(server_address: Annotated[Any, _atypes.String], name, ctx) -> Annotated[Any, _atypes.Resource]:
  server_address = _ops.convert_to_tensor(server_address, _dtypes.string)
  _inputs_flat = [server_address]
  _attrs = None
  _result = _execute.execute(b"RpcServer", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "RpcServer", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('rpc_server_register')
def rpc_server_register(server: Annotated[Any, _atypes.Resource], method_name: Annotated[Any, _atypes.String], captured_inputs, f, output_specs: str, input_specs:str="", name=None):
  r"""TODO: add doc.

  Args:
    server: A `Tensor` of type `resource`.
    method_name: A `Tensor` of type `string`.
    captured_inputs: A list of `Tensor` objects.
    f: A function decorated with @Defun.
    output_specs: A `string`.
    input_specs: An optional `string`. Defaults to `""`.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "RpcServerRegister", name, server, method_name, captured_inputs,
        "f", f, "input_specs", input_specs, "output_specs", output_specs)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_rpc_server_register(
          (server, method_name, captured_inputs, f, output_specs, input_specs,
          name,), None)
      if _result is not NotImplemented:
        return _result
      return rpc_server_register_eager_fallback(
          server, method_name, captured_inputs, f=f, input_specs=input_specs,
          output_specs=output_specs, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            rpc_server_register, (), dict(server=server,
                                          method_name=method_name,
                                          captured_inputs=captured_inputs,
                                          f=f, output_specs=output_specs,
                                          input_specs=input_specs, name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_rpc_server_register(
        (server, method_name, captured_inputs, f, output_specs, input_specs,
        name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  output_specs = _execute.make_str(output_specs, "output_specs")
  if input_specs is None:
    input_specs = ""
  input_specs = _execute.make_str(input_specs, "input_specs")
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "RpcServerRegister", server=server, method_name=method_name,
                             captured_inputs=captured_inputs, f=f,
                             output_specs=output_specs,
                             input_specs=input_specs, name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          rpc_server_register, (), dict(server=server,
                                        method_name=method_name,
                                        captured_inputs=captured_inputs, f=f,
                                        output_specs=output_specs,
                                        input_specs=input_specs, name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  return _op
RpcServerRegister = tf_export("raw_ops.RpcServerRegister")(_ops.to_raw_op(rpc_server_register))
_dispatcher_for_rpc_server_register = rpc_server_register._tf_type_based_dispatcher.Dispatch


def rpc_server_register_eager_fallback(server: Annotated[Any, _atypes.Resource], method_name: Annotated[Any, _atypes.String], captured_inputs, f, output_specs: str, input_specs: str, name, ctx):
  output_specs = _execute.make_str(output_specs, "output_specs")
  if input_specs is None:
    input_specs = ""
  input_specs = _execute.make_str(input_specs, "input_specs")
  _attr_Tin, captured_inputs = _execute.convert_to_mixed_eager_tensors(captured_inputs, ctx)
  server = _ops.convert_to_tensor(server, _dtypes.resource)
  method_name = _ops.convert_to_tensor(method_name, _dtypes.string)
  _inputs_flat = [server, method_name] + list(captured_inputs)
  _attrs = ("Tin", _attr_Tin, "f", f, "input_specs", input_specs,
  "output_specs", output_specs)
  _result = _execute.execute(b"RpcServerRegister", 0, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  _result = None
  return _result


@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('rpc_server_start')
def rpc_server_start(server: Annotated[Any, _atypes.Resource], name=None):
  r"""TODO: add doc.

  Args:
    server: A `Tensor` of type `resource`.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "RpcServerStart", name, server)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_rpc_server_start(
          (server, name,), None)
      if _result is not NotImplemented:
        return _result
      return rpc_server_start_eager_fallback(
          server, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            rpc_server_start, (), dict(server=server, name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_rpc_server_start(
        (server, name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "RpcServerStart", server=server, name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          rpc_server_start, (), dict(server=server, name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  return _op
RpcServerStart = tf_export("raw_ops.RpcServerStart")(_ops.to_raw_op(rpc_server_start))
_dispatcher_for_rpc_server_start = rpc_server_start._tf_type_based_dispatcher.Dispatch


def rpc_server_start_eager_fallback(server: Annotated[Any, _atypes.Resource], name, ctx):
  server = _ops.convert_to_tensor(server, _dtypes.resource)
  _inputs_flat = [server]
  _attrs = None
  _result = _execute.execute(b"RpcServerStart", 0, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  _result = None
  return _result

