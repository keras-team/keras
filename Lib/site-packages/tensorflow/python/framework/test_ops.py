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
@tf_export('a')
def a(name=None) -> Annotated[Any, _atypes.Float32]:
  r"""TODO: add doc.

  Args:
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `float32`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "A", name)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_a(
          (name,), None)
      if _result is not NotImplemented:
        return _result
      return a_eager_fallback(
          name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            a, (), dict(name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_a(
        (name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "A", name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          a, (), dict(name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ()
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "A", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

A = tf_export("raw_ops.A")(_ops.to_raw_op(a))
_dispatcher_for_a = a._tf_type_based_dispatcher.Dispatch


def a_eager_fallback(name, ctx) -> Annotated[Any, _atypes.Float32]:
  _inputs_flat = []
  _attrs = None
  _result = _execute.execute(b"A", 1, inputs=_inputs_flat, attrs=_attrs,
                             ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "A", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('attr')
def attr(a: int, name=None):
  r"""TODO: add doc.

  Args:
    a: An `int`.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "Attr", name, "a", a)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_attr(
          (a, name,), None)
      if _result is not NotImplemented:
        return _result
      return attr_eager_fallback(
          a=a, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            attr, (), dict(a=a, name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_attr(
        (a, name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  a = _execute.make_int(a, "a")
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "Attr", a=a, name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          attr, (), dict(a=a, name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  return _op
Attr = tf_export("raw_ops.Attr")(_ops.to_raw_op(attr))
_dispatcher_for_attr = attr._tf_type_based_dispatcher.Dispatch


def attr_eager_fallback(a: int, name, ctx):
  a = _execute.make_int(a, "a")
  _inputs_flat = []
  _attrs = ("a", a)
  _result = _execute.execute(b"Attr", 0, inputs=_inputs_flat, attrs=_attrs,
                             ctx=ctx, name=name)
  _result = None
  return _result


@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('attr_bool')
def attr_bool(a: bool, name=None):
  r"""TODO: add doc.

  Args:
    a: A `bool`.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "AttrBool", name, "a", a)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_attr_bool(
          (a, name,), None)
      if _result is not NotImplemented:
        return _result
      return attr_bool_eager_fallback(
          a=a, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            attr_bool, (), dict(a=a, name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_attr_bool(
        (a, name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  a = _execute.make_bool(a, "a")
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "AttrBool", a=a, name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          attr_bool, (), dict(a=a, name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  return _op
AttrBool = tf_export("raw_ops.AttrBool")(_ops.to_raw_op(attr_bool))
_dispatcher_for_attr_bool = attr_bool._tf_type_based_dispatcher.Dispatch


def attr_bool_eager_fallback(a: bool, name, ctx):
  a = _execute.make_bool(a, "a")
  _inputs_flat = []
  _attrs = ("a", a)
  _result = _execute.execute(b"AttrBool", 0, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  _result = None
  return _result


@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('attr_bool_list')
def attr_bool_list(a, name=None):
  r"""TODO: add doc.

  Args:
    a: A list of `bools`.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "AttrBoolList", name, "a", a)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_attr_bool_list(
          (a, name,), None)
      if _result is not NotImplemented:
        return _result
      return attr_bool_list_eager_fallback(
          a=a, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            attr_bool_list, (), dict(a=a, name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_attr_bool_list(
        (a, name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  if not isinstance(a, (list, tuple)):
    raise TypeError(
        "Expected list for 'a' argument to "
        "'attr_bool_list' Op, not %r." % a)
  a = [_execute.make_bool(_b, "a") for _b in a]
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "AttrBoolList", a=a, name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          attr_bool_list, (), dict(a=a, name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  return _op
AttrBoolList = tf_export("raw_ops.AttrBoolList")(_ops.to_raw_op(attr_bool_list))
_dispatcher_for_attr_bool_list = attr_bool_list._tf_type_based_dispatcher.Dispatch


def attr_bool_list_eager_fallback(a, name, ctx):
  if not isinstance(a, (list, tuple)):
    raise TypeError(
        "Expected list for 'a' argument to "
        "'attr_bool_list' Op, not %r." % a)
  a = [_execute.make_bool(_b, "a") for _b in a]
  _inputs_flat = []
  _attrs = ("a", a)
  _result = _execute.execute(b"AttrBoolList", 0, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  _result = None
  return _result


@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('attr_default')
def attr_default(a:str="banana", name=None):
  r"""TODO: add doc.

  Args:
    a: An optional `string`. Defaults to `"banana"`.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "AttrDefault", name, "a", a)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_attr_default(
          (a, name,), None)
      if _result is not NotImplemented:
        return _result
      return attr_default_eager_fallback(
          a=a, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            attr_default, (), dict(a=a, name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_attr_default(
        (a, name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  if a is None:
    a = "banana"
  a = _execute.make_str(a, "a")
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "AttrDefault", a=a, name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          attr_default, (), dict(a=a, name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  return _op
AttrDefault = tf_export("raw_ops.AttrDefault")(_ops.to_raw_op(attr_default))
_dispatcher_for_attr_default = attr_default._tf_type_based_dispatcher.Dispatch


def attr_default_eager_fallback(a: str, name, ctx):
  if a is None:
    a = "banana"
  a = _execute.make_str(a, "a")
  _inputs_flat = []
  _attrs = ("a", a)
  _result = _execute.execute(b"AttrDefault", 0, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  _result = None
  return _result


@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('attr_empty_list_default')
def attr_empty_list_default(a=[], name=None):
  r"""TODO: add doc.

  Args:
    a: An optional list of `floats`. Defaults to `[]`.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "AttrEmptyListDefault", name, "a", a)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_attr_empty_list_default(
          (a, name,), None)
      if _result is not NotImplemented:
        return _result
      return attr_empty_list_default_eager_fallback(
          a=a, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            attr_empty_list_default, (), dict(a=a, name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_attr_empty_list_default(
        (a, name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  if a is None:
    a = []
  if not isinstance(a, (list, tuple)):
    raise TypeError(
        "Expected list for 'a' argument to "
        "'attr_empty_list_default' Op, not %r." % a)
  a = [_execute.make_float(_f, "a") for _f in a]
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "AttrEmptyListDefault", a=a, name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          attr_empty_list_default, (), dict(a=a, name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  return _op
AttrEmptyListDefault = tf_export("raw_ops.AttrEmptyListDefault")(_ops.to_raw_op(attr_empty_list_default))
_dispatcher_for_attr_empty_list_default = attr_empty_list_default._tf_type_based_dispatcher.Dispatch


def attr_empty_list_default_eager_fallback(a, name, ctx):
  if a is None:
    a = []
  if not isinstance(a, (list, tuple)):
    raise TypeError(
        "Expected list for 'a' argument to "
        "'attr_empty_list_default' Op, not %r." % a)
  a = [_execute.make_float(_f, "a") for _f in a]
  _inputs_flat = []
  _attrs = ("a", a)
  _result = _execute.execute(b"AttrEmptyListDefault", 0, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  _result = None
  return _result


@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('attr_enum')
def attr_enum(a: str, name=None):
  r"""TODO: add doc.

  Args:
    a: A `string` from: `"apples", "oranges"`.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "AttrEnum", name, "a", a)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_attr_enum(
          (a, name,), None)
      if _result is not NotImplemented:
        return _result
      return attr_enum_eager_fallback(
          a=a, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            attr_enum, (), dict(a=a, name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_attr_enum(
        (a, name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  a = _execute.make_str(a, "a")
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "AttrEnum", a=a, name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          attr_enum, (), dict(a=a, name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  return _op
AttrEnum = tf_export("raw_ops.AttrEnum")(_ops.to_raw_op(attr_enum))
_dispatcher_for_attr_enum = attr_enum._tf_type_based_dispatcher.Dispatch


def attr_enum_eager_fallback(a: str, name, ctx):
  a = _execute.make_str(a, "a")
  _inputs_flat = []
  _attrs = ("a", a)
  _result = _execute.execute(b"AttrEnum", 0, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  _result = None
  return _result


@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('attr_enum_list')
def attr_enum_list(a, name=None):
  r"""TODO: add doc.

  Args:
    a: A list of `strings` from: `"apples", "oranges"`.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "AttrEnumList", name, "a", a)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_attr_enum_list(
          (a, name,), None)
      if _result is not NotImplemented:
        return _result
      return attr_enum_list_eager_fallback(
          a=a, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            attr_enum_list, (), dict(a=a, name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_attr_enum_list(
        (a, name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  if not isinstance(a, (list, tuple)):
    raise TypeError(
        "Expected list for 'a' argument to "
        "'attr_enum_list' Op, not %r." % a)
  a = [_execute.make_str(_s, "a") for _s in a]
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "AttrEnumList", a=a, name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          attr_enum_list, (), dict(a=a, name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  return _op
AttrEnumList = tf_export("raw_ops.AttrEnumList")(_ops.to_raw_op(attr_enum_list))
_dispatcher_for_attr_enum_list = attr_enum_list._tf_type_based_dispatcher.Dispatch


def attr_enum_list_eager_fallback(a, name, ctx):
  if not isinstance(a, (list, tuple)):
    raise TypeError(
        "Expected list for 'a' argument to "
        "'attr_enum_list' Op, not %r." % a)
  a = [_execute.make_str(_s, "a") for _s in a]
  _inputs_flat = []
  _attrs = ("a", a)
  _result = _execute.execute(b"AttrEnumList", 0, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  _result = None
  return _result


@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('attr_float')
def attr_float(a: float, name=None):
  r"""TODO: add doc.

  Args:
    a: A `float`.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "AttrFloat", name, "a", a)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_attr_float(
          (a, name,), None)
      if _result is not NotImplemented:
        return _result
      return attr_float_eager_fallback(
          a=a, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            attr_float, (), dict(a=a, name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_attr_float(
        (a, name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  a = _execute.make_float(a, "a")
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "AttrFloat", a=a, name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          attr_float, (), dict(a=a, name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  return _op
AttrFloat = tf_export("raw_ops.AttrFloat")(_ops.to_raw_op(attr_float))
_dispatcher_for_attr_float = attr_float._tf_type_based_dispatcher.Dispatch


def attr_float_eager_fallback(a: float, name, ctx):
  a = _execute.make_float(a, "a")
  _inputs_flat = []
  _attrs = ("a", a)
  _result = _execute.execute(b"AttrFloat", 0, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  _result = None
  return _result


@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('attr_list_default')
def attr_list_default(a=[5, 15], name=None):
  r"""TODO: add doc.

  Args:
    a: An optional list of `ints`. Defaults to `[5, 15]`.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "AttrListDefault", name, "a", a)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_attr_list_default(
          (a, name,), None)
      if _result is not NotImplemented:
        return _result
      return attr_list_default_eager_fallback(
          a=a, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            attr_list_default, (), dict(a=a, name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_attr_list_default(
        (a, name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  if a is None:
    a = [5, 15]
  if not isinstance(a, (list, tuple)):
    raise TypeError(
        "Expected list for 'a' argument to "
        "'attr_list_default' Op, not %r." % a)
  a = [_execute.make_int(_i, "a") for _i in a]
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "AttrListDefault", a=a, name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          attr_list_default, (), dict(a=a, name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  return _op
AttrListDefault = tf_export("raw_ops.AttrListDefault")(_ops.to_raw_op(attr_list_default))
_dispatcher_for_attr_list_default = attr_list_default._tf_type_based_dispatcher.Dispatch


def attr_list_default_eager_fallback(a, name, ctx):
  if a is None:
    a = [5, 15]
  if not isinstance(a, (list, tuple)):
    raise TypeError(
        "Expected list for 'a' argument to "
        "'attr_list_default' Op, not %r." % a)
  a = [_execute.make_int(_i, "a") for _i in a]
  _inputs_flat = []
  _attrs = ("a", a)
  _result = _execute.execute(b"AttrListDefault", 0, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  _result = None
  return _result


@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('attr_list_min')
def attr_list_min(a, name=None):
  r"""TODO: add doc.

  Args:
    a: A list of `ints` that has length `>= 2`.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "AttrListMin", name, "a", a)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_attr_list_min(
          (a, name,), None)
      if _result is not NotImplemented:
        return _result
      return attr_list_min_eager_fallback(
          a=a, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            attr_list_min, (), dict(a=a, name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_attr_list_min(
        (a, name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  if not isinstance(a, (list, tuple)):
    raise TypeError(
        "Expected list for 'a' argument to "
        "'attr_list_min' Op, not %r." % a)
  a = [_execute.make_int(_i, "a") for _i in a]
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "AttrListMin", a=a, name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          attr_list_min, (), dict(a=a, name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  return _op
AttrListMin = tf_export("raw_ops.AttrListMin")(_ops.to_raw_op(attr_list_min))
_dispatcher_for_attr_list_min = attr_list_min._tf_type_based_dispatcher.Dispatch


def attr_list_min_eager_fallback(a, name, ctx):
  if not isinstance(a, (list, tuple)):
    raise TypeError(
        "Expected list for 'a' argument to "
        "'attr_list_min' Op, not %r." % a)
  a = [_execute.make_int(_i, "a") for _i in a]
  _inputs_flat = []
  _attrs = ("a", a)
  _result = _execute.execute(b"AttrListMin", 0, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  _result = None
  return _result


TV_AttrListTypeDefault_T = TypeVar("TV_AttrListTypeDefault_T", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float16, _atypes.Float32, _atypes.Float64, _atypes.Float8e4m3fn, _atypes.Float8e5m2, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int4, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.Resource, _atypes.String, _atypes.UInt16, _atypes.UInt32, _atypes.UInt4, _atypes.UInt64, _atypes.UInt8, _atypes.Variant)

@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('attr_list_type_default')
def attr_list_type_default(a: Annotated[List[Any], TV_AttrListTypeDefault_T], b: Annotated[List[Any], TV_AttrListTypeDefault_T], name=None):
  r"""TODO: add doc.

  Args:
    a: A list of at least 1 `Tensor` objects with the same type.
    b: A list with the same length as `a` of `Tensor` objects with the same type as `a`.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "AttrListTypeDefault", name, a, b)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_attr_list_type_default(
          (a, b, name,), None)
      if _result is not NotImplemented:
        return _result
      return attr_list_type_default_eager_fallback(
          a, b, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            attr_list_type_default, (), dict(a=a, b=b, name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_attr_list_type_default(
        (a, b, name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  if not isinstance(a, (list, tuple)):
    raise TypeError(
        "Expected list for 'a' argument to "
        "'attr_list_type_default' Op, not %r." % a)
  _attr_N = len(a)
  if not isinstance(b, (list, tuple)):
    raise TypeError(
        "Expected list for 'b' argument to "
        "'attr_list_type_default' Op, not %r." % b)
  if len(b) != _attr_N:
    raise ValueError(
        "List argument 'b' to 'attr_list_type_default' Op with length %d "
        "must match length %d of argument 'a'." %
        (len(b), _attr_N))
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "AttrListTypeDefault", a=a, b=b, name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          attr_list_type_default, (), dict(a=a, b=b, name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  return _op
AttrListTypeDefault = tf_export("raw_ops.AttrListTypeDefault")(_ops.to_raw_op(attr_list_type_default))
_dispatcher_for_attr_list_type_default = attr_list_type_default._tf_type_based_dispatcher.Dispatch


def attr_list_type_default_eager_fallback(a: Annotated[List[Any], TV_AttrListTypeDefault_T], b: Annotated[List[Any], TV_AttrListTypeDefault_T], name, ctx):
  if not isinstance(a, (list, tuple)):
    raise TypeError(
        "Expected list for 'a' argument to "
        "'attr_list_type_default' Op, not %r." % a)
  _attr_N = len(a)
  if not isinstance(b, (list, tuple)):
    raise TypeError(
        "Expected list for 'b' argument to "
        "'attr_list_type_default' Op, not %r." % b)
  if len(b) != _attr_N:
    raise ValueError(
        "List argument 'b' to 'attr_list_type_default' Op with length %d "
        "must match length %d of argument 'a'." %
        (len(b), _attr_N))
  _attr_T, _inputs_T = _execute.args_to_matching_eager(list(a) + list(b), ctx, [], _dtypes.int32)
  _inputs_T = [_inputs_T[:_attr_N]] + _inputs_T[_attr_N:]
  _inputs_T = _inputs_T[:1] + [_inputs_T[1:]]
  (a, b) = _inputs_T
  _inputs_flat = list(a) + list(b)
  _attrs = ("T", _attr_T, "N", _attr_N)
  _result = _execute.execute(b"AttrListTypeDefault", 0, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  _result = None
  return _result


@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('attr_min')
def attr_min(a: int, name=None):
  r"""TODO: add doc.

  Args:
    a: An `int` that is `>= 5`.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "AttrMin", name, "a", a)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_attr_min(
          (a, name,), None)
      if _result is not NotImplemented:
        return _result
      return attr_min_eager_fallback(
          a=a, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            attr_min, (), dict(a=a, name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_attr_min(
        (a, name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  a = _execute.make_int(a, "a")
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "AttrMin", a=a, name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          attr_min, (), dict(a=a, name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  return _op
AttrMin = tf_export("raw_ops.AttrMin")(_ops.to_raw_op(attr_min))
_dispatcher_for_attr_min = attr_min._tf_type_based_dispatcher.Dispatch


def attr_min_eager_fallback(a: int, name, ctx):
  a = _execute.make_int(a, "a")
  _inputs_flat = []
  _attrs = ("a", a)
  _result = _execute.execute(b"AttrMin", 0, inputs=_inputs_flat, attrs=_attrs,
                             ctx=ctx, name=name)
  _result = None
  return _result


@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('attr_partial_shape')
def attr_partial_shape(a, name=None):
  r"""TODO: add doc.

  Args:
    a: A `tf.TensorShape` or list of `ints`.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "AttrPartialShape", name, "a", a)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_attr_partial_shape(
          (a, name,), None)
      if _result is not NotImplemented:
        return _result
      return attr_partial_shape_eager_fallback(
          a=a, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            attr_partial_shape, (), dict(a=a, name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_attr_partial_shape(
        (a, name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  a = _execute.make_shape(a, "a")
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "AttrPartialShape", a=a, name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          attr_partial_shape, (), dict(a=a, name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  return _op
AttrPartialShape = tf_export("raw_ops.AttrPartialShape")(_ops.to_raw_op(attr_partial_shape))
_dispatcher_for_attr_partial_shape = attr_partial_shape._tf_type_based_dispatcher.Dispatch


def attr_partial_shape_eager_fallback(a, name, ctx):
  a = _execute.make_shape(a, "a")
  _inputs_flat = []
  _attrs = ("a", a)
  _result = _execute.execute(b"AttrPartialShape", 0, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  _result = None
  return _result


@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('attr_partial_shape_list')
def attr_partial_shape_list(a, name=None):
  r"""TODO: add doc.

  Args:
    a: A list of shapes (each a `tf.TensorShape` or list of `ints`).
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "AttrPartialShapeList", name, "a", a)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_attr_partial_shape_list(
          (a, name,), None)
      if _result is not NotImplemented:
        return _result
      return attr_partial_shape_list_eager_fallback(
          a=a, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            attr_partial_shape_list, (), dict(a=a, name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_attr_partial_shape_list(
        (a, name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  if not isinstance(a, (list, tuple)):
    raise TypeError(
        "Expected list for 'a' argument to "
        "'attr_partial_shape_list' Op, not %r." % a)
  a = [_execute.make_shape(_s, "a") for _s in a]
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "AttrPartialShapeList", a=a, name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          attr_partial_shape_list, (), dict(a=a, name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  return _op
AttrPartialShapeList = tf_export("raw_ops.AttrPartialShapeList")(_ops.to_raw_op(attr_partial_shape_list))
_dispatcher_for_attr_partial_shape_list = attr_partial_shape_list._tf_type_based_dispatcher.Dispatch


def attr_partial_shape_list_eager_fallback(a, name, ctx):
  if not isinstance(a, (list, tuple)):
    raise TypeError(
        "Expected list for 'a' argument to "
        "'attr_partial_shape_list' Op, not %r." % a)
  a = [_execute.make_shape(_s, "a") for _s in a]
  _inputs_flat = []
  _attrs = ("a", a)
  _result = _execute.execute(b"AttrPartialShapeList", 0, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  _result = None
  return _result


@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('attr_shape')
def attr_shape(a, name=None):
  r"""TODO: add doc.

  Args:
    a: A `tf.TensorShape` or list of `ints`.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "AttrShape", name, "a", a)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_attr_shape(
          (a, name,), None)
      if _result is not NotImplemented:
        return _result
      return attr_shape_eager_fallback(
          a=a, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            attr_shape, (), dict(a=a, name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_attr_shape(
        (a, name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  a = _execute.make_shape(a, "a")
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "AttrShape", a=a, name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          attr_shape, (), dict(a=a, name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  return _op
AttrShape = tf_export("raw_ops.AttrShape")(_ops.to_raw_op(attr_shape))
_dispatcher_for_attr_shape = attr_shape._tf_type_based_dispatcher.Dispatch


def attr_shape_eager_fallback(a, name, ctx):
  a = _execute.make_shape(a, "a")
  _inputs_flat = []
  _attrs = ("a", a)
  _result = _execute.execute(b"AttrShape", 0, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  _result = None
  return _result


@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('attr_shape_list')
def attr_shape_list(a, name=None):
  r"""TODO: add doc.

  Args:
    a: A list of shapes (each a `tf.TensorShape` or list of `ints`).
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "AttrShapeList", name, "a", a)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_attr_shape_list(
          (a, name,), None)
      if _result is not NotImplemented:
        return _result
      return attr_shape_list_eager_fallback(
          a=a, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            attr_shape_list, (), dict(a=a, name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_attr_shape_list(
        (a, name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  if not isinstance(a, (list, tuple)):
    raise TypeError(
        "Expected list for 'a' argument to "
        "'attr_shape_list' Op, not %r." % a)
  a = [_execute.make_shape(_s, "a") for _s in a]
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "AttrShapeList", a=a, name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          attr_shape_list, (), dict(a=a, name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  return _op
AttrShapeList = tf_export("raw_ops.AttrShapeList")(_ops.to_raw_op(attr_shape_list))
_dispatcher_for_attr_shape_list = attr_shape_list._tf_type_based_dispatcher.Dispatch


def attr_shape_list_eager_fallback(a, name, ctx):
  if not isinstance(a, (list, tuple)):
    raise TypeError(
        "Expected list for 'a' argument to "
        "'attr_shape_list' Op, not %r." % a)
  a = [_execute.make_shape(_s, "a") for _s in a]
  _inputs_flat = []
  _attrs = ("a", a)
  _result = _execute.execute(b"AttrShapeList", 0, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  _result = None
  return _result


TV_AttrTypeDefault_T = TypeVar("TV_AttrTypeDefault_T", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float16, _atypes.Float32, _atypes.Float64, _atypes.Float8e4m3fn, _atypes.Float8e5m2, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int4, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.Resource, _atypes.String, _atypes.UInt16, _atypes.UInt32, _atypes.UInt4, _atypes.UInt64, _atypes.UInt8, _atypes.Variant)

@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('attr_type_default')
def attr_type_default(a: Annotated[Any, TV_AttrTypeDefault_T], name=None):
  r"""TODO: add doc.

  Args:
    a: A `Tensor`.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "AttrTypeDefault", name, a)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_attr_type_default(
          (a, name,), None)
      if _result is not NotImplemented:
        return _result
      return attr_type_default_eager_fallback(
          a, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            attr_type_default, (), dict(a=a, name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_attr_type_default(
        (a, name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "AttrTypeDefault", a=a, name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          attr_type_default, (), dict(a=a, name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  return _op
AttrTypeDefault = tf_export("raw_ops.AttrTypeDefault")(_ops.to_raw_op(attr_type_default))
_dispatcher_for_attr_type_default = attr_type_default._tf_type_based_dispatcher.Dispatch


def attr_type_default_eager_fallback(a: Annotated[Any, TV_AttrTypeDefault_T], name, ctx):
  _attr_T, (a,) = _execute.args_to_matching_eager([a], ctx, [], _dtypes.int32)
  _inputs_flat = [a]
  _attrs = ("T", _attr_T)
  _result = _execute.execute(b"AttrTypeDefault", 0, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  _result = None
  return _result


@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('b')
def b(name=None) -> Annotated[Any, _atypes.Float32]:
  r"""TODO: add doc.

  Args:
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `float32`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "B", name)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_b(
          (name,), None)
      if _result is not NotImplemented:
        return _result
      return b_eager_fallback(
          name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            b, (), dict(name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_b(
        (name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "B", name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          b, (), dict(name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ()
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "B", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

B = tf_export("raw_ops.B")(_ops.to_raw_op(b))
_dispatcher_for_b = b._tf_type_based_dispatcher.Dispatch


def b_eager_fallback(name, ctx) -> Annotated[Any, _atypes.Float32]:
  _inputs_flat = []
  _attrs = None
  _result = _execute.execute(b"B", 1, inputs=_inputs_flat, attrs=_attrs,
                             ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "B", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_Binary_T = TypeVar("TV_Binary_T", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float16, _atypes.Float32, _atypes.Float64, _atypes.Float8e4m3fn, _atypes.Float8e5m2, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int4, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.Resource, _atypes.String, _atypes.UInt16, _atypes.UInt32, _atypes.UInt4, _atypes.UInt64, _atypes.UInt8, _atypes.Variant)

@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('binary')
def binary(a: Annotated[Any, TV_Binary_T], b: Annotated[Any, TV_Binary_T], name=None) -> Annotated[Any, TV_Binary_T]:
  r"""TODO: add doc.

  Args:
    a: A `Tensor`.
    b: A `Tensor`. Must have the same type as `a`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `a`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "Binary", name, a, b)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_binary(
          (a, b, name,), None)
      if _result is not NotImplemented:
        return _result
      return binary_eager_fallback(
          a, b, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            binary, (), dict(a=a, b=b, name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_binary(
        (a, b, name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "Binary", a=a, b=b, name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          binary, (), dict(a=a, b=b, name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "Binary", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

Binary = tf_export("raw_ops.Binary")(_ops.to_raw_op(binary))
_dispatcher_for_binary = binary._tf_type_based_dispatcher.Dispatch


def binary_eager_fallback(a: Annotated[Any, TV_Binary_T], b: Annotated[Any, TV_Binary_T], name, ctx) -> Annotated[Any, TV_Binary_T]:
  _attr_T, _inputs_T = _execute.args_to_matching_eager([a, b], ctx, [])
  (a, b) = _inputs_T
  _inputs_flat = [a, b]
  _attrs = ("T", _attr_T)
  _result = _execute.execute(b"Binary", 1, inputs=_inputs_flat, attrs=_attrs,
                             ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "Binary", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

_ComplexStructOutput = collections.namedtuple(
    "ComplexStruct",
    ["a", "b", "c"])


@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('complex_struct')
def complex_struct(n_a: int, n_b: int, t_c, name=None):
  r"""TODO: add doc.

  Args:
    n_a: An `int` that is `>= 0`.
    n_b: An `int` that is `>= 0`.
    t_c: A list of `tf.DTypes`.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (a, b, c).

    a: A list of `n_a` `Tensor` objects with type `int32`.
    b: A list of `n_b` `Tensor` objects with type `int64`.
    c: A list of `Tensor` objects of type `t_c`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "ComplexStruct", name, "n_a", n_a, "n_b", n_b, "t_c", t_c)
      _result = _ComplexStructOutput._make(_result)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_complex_struct(
          (n_a, n_b, t_c, name,), None)
      if _result is not NotImplemented:
        return _result
      return complex_struct_eager_fallback(
          n_a=n_a, n_b=n_b, t_c=t_c, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            complex_struct, (), dict(n_a=n_a, n_b=n_b, t_c=t_c, name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_complex_struct(
        (n_a, n_b, t_c, name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  n_a = _execute.make_int(n_a, "n_a")
  n_b = _execute.make_int(n_b, "n_b")
  if not isinstance(t_c, (list, tuple)):
    raise TypeError(
        "Expected list for 't_c' argument to "
        "'complex_struct' Op, not %r." % t_c)
  t_c = [_execute.make_type(_t, "t_c") for _t in t_c]
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "ComplexStruct", n_a=n_a, n_b=n_b, t_c=t_c, name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          complex_struct, (), dict(n_a=n_a, n_b=n_b, t_c=t_c, name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("n_a", _op._get_attr_int("n_a"), "n_b",
              _op._get_attr_int("n_b"), "t_c", _op.get_attr("t_c"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "ComplexStruct", _inputs_flat, _attrs, _result)
  _result = [_result[:n_a]] + _result[n_a:]
  _result = _result[:1] + [_result[1:1 + n_b]] + _result[1 + n_b:]
  _result = _result[:2] + [_result[2:]]
  _result = _ComplexStructOutput._make(_result)
  return _result

ComplexStruct = tf_export("raw_ops.ComplexStruct")(_ops.to_raw_op(complex_struct))
_dispatcher_for_complex_struct = complex_struct._tf_type_based_dispatcher.Dispatch


def complex_struct_eager_fallback(n_a: int, n_b: int, t_c, name, ctx):
  n_a = _execute.make_int(n_a, "n_a")
  n_b = _execute.make_int(n_b, "n_b")
  if not isinstance(t_c, (list, tuple)):
    raise TypeError(
        "Expected list for 't_c' argument to "
        "'complex_struct' Op, not %r." % t_c)
  t_c = [_execute.make_type(_t, "t_c") for _t in t_c]
  _inputs_flat = []
  _attrs = ("n_a", n_a, "n_b", n_b, "t_c", t_c)
  _result = _execute.execute(b"ComplexStruct", n_a + n_b + len(t_c),
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "ComplexStruct", _inputs_flat, _attrs, _result)
  _result = [_result[:n_a]] + _result[n_a:]
  _result = _result[:1] + [_result[1:1 + n_b]] + _result[1 + n_b:]
  _result = _result[:2] + [_result[2:]]
  _result = _ComplexStructOutput._make(_result)
  return _result


TV_CopyOp_T = TypeVar("TV_CopyOp_T", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float16, _atypes.Float32, _atypes.Float64, _atypes.Float8e4m3fn, _atypes.Float8e5m2, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int4, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.Resource, _atypes.String, _atypes.UInt16, _atypes.UInt32, _atypes.UInt4, _atypes.UInt64, _atypes.UInt8, _atypes.Variant)

@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('copy_op')
def copy_op(a: Annotated[Any, TV_CopyOp_T], name=None) -> Annotated[Any, TV_CopyOp_T]:
  r"""TODO: add doc.

  Args:
    a: A `Tensor`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `a`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "CopyOp", name, a)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_copy_op(
          (a, name,), None)
      if _result is not NotImplemented:
        return _result
      return copy_op_eager_fallback(
          a, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            copy_op, (), dict(a=a, name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_copy_op(
        (a, name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "CopyOp", a=a, name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          copy_op, (), dict(a=a, name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "CopyOp", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

CopyOp = tf_export("raw_ops.CopyOp")(_ops.to_raw_op(copy_op))
_dispatcher_for_copy_op = copy_op._tf_type_based_dispatcher.Dispatch


def copy_op_eager_fallback(a: Annotated[Any, TV_CopyOp_T], name, ctx) -> Annotated[Any, TV_CopyOp_T]:
  _attr_T, (a,) = _execute.args_to_matching_eager([a], ctx, [])
  _inputs_flat = [a]
  _attrs = ("T", _attr_T)
  _result = _execute.execute(b"CopyOp", 1, inputs=_inputs_flat, attrs=_attrs,
                             ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "CopyOp", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_DefaultAttrs_type_val = TypeVar("TV_DefaultAttrs_type_val", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float16, _atypes.Float32, _atypes.Float64, _atypes.Float8e4m3fn, _atypes.Float8e5m2, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int4, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.Resource, _atypes.String, _atypes.UInt16, _atypes.UInt32, _atypes.UInt4, _atypes.UInt64, _atypes.UInt8, _atypes.Variant)

@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('default_attrs')
def default_attrs(string_val:str="abc", string_list_val=["abc", ""], int_val:int=123, int_list_val=[1, 2, 3], float_val:float=10, float_list_val=[10], bool_val:bool=True, bool_list_val=[True, False], type_val:TV_DefaultAttrs_type_val=_dtypes.int32, type_list_val=[_dtypes.int32, _dtypes.float32], shape_val=[2, 1], shape_list_val=[[], [1]], tensor_val=_execute.make_tensor("""dtype: DT_INT32 tensor_shape { } int_val: 1 """, "tensor_val"), tensor_list_val=[_execute.make_tensor(_pb, "tensor_list_val") for _pb in ("""dtype: DT_INT32 tensor_shape { } int_val: 1 """,)], name=None):
  r"""TODO: add doc.

  Args:
    string_val: An optional `string`. Defaults to `"abc"`.
    string_list_val: An optional list of `strings`. Defaults to `["abc", ""]`.
    int_val: An optional `int`. Defaults to `123`.
    int_list_val: An optional list of `ints`. Defaults to `[1, 2, 3]`.
    float_val: An optional `float`. Defaults to `10`.
    float_list_val: An optional list of `floats`. Defaults to `[10]`.
    bool_val: An optional `bool`. Defaults to `True`.
    bool_list_val: An optional list of `bools`. Defaults to `[True, False]`.
    type_val: An optional `tf.DType`. Defaults to `tf.int32`.
    type_list_val: An optional list of `tf.DTypes`. Defaults to `[tf.int32, tf.float32]`.
    shape_val: An optional `tf.TensorShape` or list of `ints`. Defaults to `[2, 1]`.
    shape_list_val: An optional list of shapes (each a `tf.TensorShape` or list of `ints`). Defaults to `[[], [1]]`.
    tensor_val: An optional `tf.TensorProto`. Defaults to `dtype: DT_INT32 tensor_shape { } int_val: 1`.
    tensor_list_val: An optional list of `tf.TensorProto` objects. Defaults to `[dtype: DT_INT32 tensor_shape { } int_val: 1]`.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "DefaultAttrs", name, "string_val", string_val,
        "string_list_val", string_list_val, "int_val", int_val,
        "int_list_val", int_list_val, "float_val", float_val,
        "float_list_val", float_list_val, "bool_val", bool_val,
        "bool_list_val", bool_list_val, "type_val", type_val, "type_list_val",
        type_list_val, "shape_val", shape_val, "shape_list_val",
        shape_list_val, "tensor_val", tensor_val, "tensor_list_val",
        tensor_list_val)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_default_attrs(
          (string_val, string_list_val, int_val, int_list_val, float_val,
          float_list_val, bool_val, bool_list_val, type_val, type_list_val,
          shape_val, shape_list_val, tensor_val, tensor_list_val, name,), None)
      if _result is not NotImplemented:
        return _result
      return default_attrs_eager_fallback(
          string_val=string_val, string_list_val=string_list_val,
          int_val=int_val, int_list_val=int_list_val, float_val=float_val,
          float_list_val=float_list_val, bool_val=bool_val,
          bool_list_val=bool_list_val, type_val=type_val,
          type_list_val=type_list_val, shape_val=shape_val,
          shape_list_val=shape_list_val, tensor_val=tensor_val,
          tensor_list_val=tensor_list_val, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            default_attrs, (), dict(string_val=string_val,
                                    string_list_val=string_list_val,
                                    int_val=int_val,
                                    int_list_val=int_list_val,
                                    float_val=float_val,
                                    float_list_val=float_list_val,
                                    bool_val=bool_val,
                                    bool_list_val=bool_list_val,
                                    type_val=type_val,
                                    type_list_val=type_list_val,
                                    shape_val=shape_val,
                                    shape_list_val=shape_list_val,
                                    tensor_val=tensor_val,
                                    tensor_list_val=tensor_list_val,
                                    name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_default_attrs(
        (string_val, string_list_val, int_val, int_list_val, float_val,
        float_list_val, bool_val, bool_list_val, type_val, type_list_val,
        shape_val, shape_list_val, tensor_val, tensor_list_val, name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  if string_val is None:
    string_val = "abc"
  string_val = _execute.make_str(string_val, "string_val")
  if string_list_val is None:
    string_list_val = ["abc", ""]
  if not isinstance(string_list_val, (list, tuple)):
    raise TypeError(
        "Expected list for 'string_list_val' argument to "
        "'default_attrs' Op, not %r." % string_list_val)
  string_list_val = [_execute.make_str(_s, "string_list_val") for _s in string_list_val]
  if int_val is None:
    int_val = 123
  int_val = _execute.make_int(int_val, "int_val")
  if int_list_val is None:
    int_list_val = [1, 2, 3]
  if not isinstance(int_list_val, (list, tuple)):
    raise TypeError(
        "Expected list for 'int_list_val' argument to "
        "'default_attrs' Op, not %r." % int_list_val)
  int_list_val = [_execute.make_int(_i, "int_list_val") for _i in int_list_val]
  if float_val is None:
    float_val = 10
  float_val = _execute.make_float(float_val, "float_val")
  if float_list_val is None:
    float_list_val = [10]
  if not isinstance(float_list_val, (list, tuple)):
    raise TypeError(
        "Expected list for 'float_list_val' argument to "
        "'default_attrs' Op, not %r." % float_list_val)
  float_list_val = [_execute.make_float(_f, "float_list_val") for _f in float_list_val]
  if bool_val is None:
    bool_val = True
  bool_val = _execute.make_bool(bool_val, "bool_val")
  if bool_list_val is None:
    bool_list_val = [True, False]
  if not isinstance(bool_list_val, (list, tuple)):
    raise TypeError(
        "Expected list for 'bool_list_val' argument to "
        "'default_attrs' Op, not %r." % bool_list_val)
  bool_list_val = [_execute.make_bool(_b, "bool_list_val") for _b in bool_list_val]
  if type_val is None:
    type_val = _dtypes.int32
  type_val = _execute.make_type(type_val, "type_val")
  if type_list_val is None:
    type_list_val = [_dtypes.int32, _dtypes.float32]
  if not isinstance(type_list_val, (list, tuple)):
    raise TypeError(
        "Expected list for 'type_list_val' argument to "
        "'default_attrs' Op, not %r." % type_list_val)
  type_list_val = [_execute.make_type(_t, "type_list_val") for _t in type_list_val]
  if shape_val is None:
    shape_val = [2, 1]
  shape_val = _execute.make_shape(shape_val, "shape_val")
  if shape_list_val is None:
    shape_list_val = [[], [1]]
  if not isinstance(shape_list_val, (list, tuple)):
    raise TypeError(
        "Expected list for 'shape_list_val' argument to "
        "'default_attrs' Op, not %r." % shape_list_val)
  shape_list_val = [_execute.make_shape(_s, "shape_list_val") for _s in shape_list_val]
  if tensor_val is None:
    tensor_val = _execute.make_tensor("""dtype: DT_INT32 tensor_shape { } int_val: 1 """, "tensor_val")
  tensor_val = _execute.make_tensor(tensor_val, "tensor_val")
  if tensor_list_val is None:
    tensor_list_val = [_execute.make_tensor(_pb, "tensor_list_val") for _pb in ("""dtype: DT_INT32 tensor_shape { } int_val: 1 """,)]
  if not isinstance(tensor_list_val, (list, tuple)):
    raise TypeError(
        "Expected list for 'tensor_list_val' argument to "
        "'default_attrs' Op, not %r." % tensor_list_val)
  tensor_list_val = [_execute.make_tensor(_t, "tensor_list_val") for _t in tensor_list_val]
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "DefaultAttrs", string_val=string_val,
                        string_list_val=string_list_val, int_val=int_val,
                        int_list_val=int_list_val, float_val=float_val,
                        float_list_val=float_list_val, bool_val=bool_val,
                        bool_list_val=bool_list_val, type_val=type_val,
                        type_list_val=type_list_val, shape_val=shape_val,
                        shape_list_val=shape_list_val, tensor_val=tensor_val,
                        tensor_list_val=tensor_list_val, name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          default_attrs, (), dict(string_val=string_val,
                                  string_list_val=string_list_val,
                                  int_val=int_val, int_list_val=int_list_val,
                                  float_val=float_val,
                                  float_list_val=float_list_val,
                                  bool_val=bool_val,
                                  bool_list_val=bool_list_val,
                                  type_val=type_val,
                                  type_list_val=type_list_val,
                                  shape_val=shape_val,
                                  shape_list_val=shape_list_val,
                                  tensor_val=tensor_val,
                                  tensor_list_val=tensor_list_val, name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  return _op
DefaultAttrs = tf_export("raw_ops.DefaultAttrs")(_ops.to_raw_op(default_attrs))
_dispatcher_for_default_attrs = default_attrs._tf_type_based_dispatcher.Dispatch


def default_attrs_eager_fallback(string_val: str, string_list_val, int_val: int, int_list_val, float_val: float, float_list_val, bool_val: bool, bool_list_val, type_val: TV_DefaultAttrs_type_val, type_list_val, shape_val, shape_list_val, tensor_val, tensor_list_val, name, ctx):
  if string_val is None:
    string_val = "abc"
  string_val = _execute.make_str(string_val, "string_val")
  if string_list_val is None:
    string_list_val = ["abc", ""]
  if not isinstance(string_list_val, (list, tuple)):
    raise TypeError(
        "Expected list for 'string_list_val' argument to "
        "'default_attrs' Op, not %r." % string_list_val)
  string_list_val = [_execute.make_str(_s, "string_list_val") for _s in string_list_val]
  if int_val is None:
    int_val = 123
  int_val = _execute.make_int(int_val, "int_val")
  if int_list_val is None:
    int_list_val = [1, 2, 3]
  if not isinstance(int_list_val, (list, tuple)):
    raise TypeError(
        "Expected list for 'int_list_val' argument to "
        "'default_attrs' Op, not %r." % int_list_val)
  int_list_val = [_execute.make_int(_i, "int_list_val") for _i in int_list_val]
  if float_val is None:
    float_val = 10
  float_val = _execute.make_float(float_val, "float_val")
  if float_list_val is None:
    float_list_val = [10]
  if not isinstance(float_list_val, (list, tuple)):
    raise TypeError(
        "Expected list for 'float_list_val' argument to "
        "'default_attrs' Op, not %r." % float_list_val)
  float_list_val = [_execute.make_float(_f, "float_list_val") for _f in float_list_val]
  if bool_val is None:
    bool_val = True
  bool_val = _execute.make_bool(bool_val, "bool_val")
  if bool_list_val is None:
    bool_list_val = [True, False]
  if not isinstance(bool_list_val, (list, tuple)):
    raise TypeError(
        "Expected list for 'bool_list_val' argument to "
        "'default_attrs' Op, not %r." % bool_list_val)
  bool_list_val = [_execute.make_bool(_b, "bool_list_val") for _b in bool_list_val]
  if type_val is None:
    type_val = _dtypes.int32
  type_val = _execute.make_type(type_val, "type_val")
  if type_list_val is None:
    type_list_val = [_dtypes.int32, _dtypes.float32]
  if not isinstance(type_list_val, (list, tuple)):
    raise TypeError(
        "Expected list for 'type_list_val' argument to "
        "'default_attrs' Op, not %r." % type_list_val)
  type_list_val = [_execute.make_type(_t, "type_list_val") for _t in type_list_val]
  if shape_val is None:
    shape_val = [2, 1]
  shape_val = _execute.make_shape(shape_val, "shape_val")
  if shape_list_val is None:
    shape_list_val = [[], [1]]
  if not isinstance(shape_list_val, (list, tuple)):
    raise TypeError(
        "Expected list for 'shape_list_val' argument to "
        "'default_attrs' Op, not %r." % shape_list_val)
  shape_list_val = [_execute.make_shape(_s, "shape_list_val") for _s in shape_list_val]
  if tensor_val is None:
    tensor_val = _execute.make_tensor("""dtype: DT_INT32 tensor_shape { } int_val: 1 """, "tensor_val")
  tensor_val = _execute.make_tensor(tensor_val, "tensor_val")
  if tensor_list_val is None:
    tensor_list_val = [_execute.make_tensor(_pb, "tensor_list_val") for _pb in ("""dtype: DT_INT32 tensor_shape { } int_val: 1 """,)]
  if not isinstance(tensor_list_val, (list, tuple)):
    raise TypeError(
        "Expected list for 'tensor_list_val' argument to "
        "'default_attrs' Op, not %r." % tensor_list_val)
  tensor_list_val = [_execute.make_tensor(_t, "tensor_list_val") for _t in tensor_list_val]
  _inputs_flat = []
  _attrs = ("string_val", string_val, "string_list_val", string_list_val,
  "int_val", int_val, "int_list_val", int_list_val, "float_val", float_val,
  "float_list_val", float_list_val, "bool_val", bool_val, "bool_list_val",
  bool_list_val, "type_val", type_val, "type_list_val", type_list_val,
  "shape_val", shape_val, "shape_list_val", shape_list_val, "tensor_val",
  tensor_val, "tensor_list_val", tensor_list_val)
  _result = _execute.execute(b"DefaultAttrs", 0, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  _result = None
  return _result


@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('device_placement_op')
def device_placement_op(name=None) -> Annotated[Any, _atypes.String]:
  r"""TODO: add doc.

  Args:
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `string`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "DevicePlacementOp", name)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_device_placement_op(
          (name,), None)
      if _result is not NotImplemented:
        return _result
      return device_placement_op_eager_fallback(
          name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            device_placement_op, (), dict(name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_device_placement_op(
        (name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "DevicePlacementOp", name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          device_placement_op, (), dict(name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ()
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "DevicePlacementOp", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

DevicePlacementOp = tf_export("raw_ops.DevicePlacementOp")(_ops.to_raw_op(device_placement_op))
_dispatcher_for_device_placement_op = device_placement_op._tf_type_based_dispatcher.Dispatch


def device_placement_op_eager_fallback(name, ctx) -> Annotated[Any, _atypes.String]:
  _inputs_flat = []
  _attrs = None
  _result = _execute.execute(b"DevicePlacementOp", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "DevicePlacementOp", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_DtypeWithDefaultOp_T = TypeVar("TV_DtypeWithDefaultOp_T", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float16, _atypes.Float32, _atypes.Float64, _atypes.Float8e4m3fn, _atypes.Float8e5m2, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int4, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.Resource, _atypes.String, _atypes.UInt16, _atypes.UInt32, _atypes.UInt4, _atypes.UInt64, _atypes.UInt8, _atypes.Variant)

@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('dtype_with_default_op')
def dtype_with_default_op(in_: Annotated[Any, TV_DtypeWithDefaultOp_T], name=None) -> Annotated[Any, _atypes.String]:
  r"""TODO: add doc.

  Args:
    in_: A `Tensor`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `string`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "DtypeWithDefaultOp", name, in_)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_dtype_with_default_op(
          (in_, name,), None)
      if _result is not NotImplemented:
        return _result
      return dtype_with_default_op_eager_fallback(
          in_, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            dtype_with_default_op, (), dict(in_=in_, name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_dtype_with_default_op(
        (in_, name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "DtypeWithDefaultOp", in_=in_, name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          dtype_with_default_op, (), dict(in_=in_, name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "DtypeWithDefaultOp", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

DtypeWithDefaultOp = tf_export("raw_ops.DtypeWithDefaultOp")(_ops.to_raw_op(dtype_with_default_op))
_dispatcher_for_dtype_with_default_op = dtype_with_default_op._tf_type_based_dispatcher.Dispatch


def dtype_with_default_op_eager_fallback(in_: Annotated[Any, TV_DtypeWithDefaultOp_T], name, ctx) -> Annotated[Any, _atypes.String]:
  _attr_T, (in_,) = _execute.args_to_matching_eager([in_], ctx, [], _dtypes.uint8)
  _inputs_flat = [in_]
  _attrs = ("T", _attr_T)
  _result = _execute.execute(b"DtypeWithDefaultOp", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "DtypeWithDefaultOp", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

_FiveFloatOutputsOutput = collections.namedtuple(
    "FiveFloatOutputs",
    ["a", "b", "c", "d", "e"])


@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('five_float_outputs')
def five_float_outputs(name=None):
  r"""TODO: add doc.

  Args:
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (a, b, c, d, e).

    a: A `Tensor` of type `float32`.
    b: A `Tensor` of type `float32`.
    c: A `Tensor` of type `float32`.
    d: A `Tensor` of type `float32`.
    e: A `Tensor` of type `float32`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "FiveFloatOutputs", name)
      _result = _FiveFloatOutputsOutput._make(_result)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_five_float_outputs(
          (name,), None)
      if _result is not NotImplemented:
        return _result
      return five_float_outputs_eager_fallback(
          name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            five_float_outputs, (), dict(name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_five_float_outputs(
        (name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "FiveFloatOutputs", name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          five_float_outputs, (), dict(name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ()
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "FiveFloatOutputs", _inputs_flat, _attrs, _result)
  _result = _FiveFloatOutputsOutput._make(_result)
  return _result

FiveFloatOutputs = tf_export("raw_ops.FiveFloatOutputs")(_ops.to_raw_op(five_float_outputs))
_dispatcher_for_five_float_outputs = five_float_outputs._tf_type_based_dispatcher.Dispatch


def five_float_outputs_eager_fallback(name, ctx):
  _inputs_flat = []
  _attrs = None
  _result = _execute.execute(b"FiveFloatOutputs", 5, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "FiveFloatOutputs", _inputs_flat, _attrs, _result)
  _result = _FiveFloatOutputsOutput._make(_result)
  return _result


@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('float_input')
def float_input(a: Annotated[Any, _atypes.Float32], name=None):
  r"""TODO: add doc.

  Args:
    a: A `Tensor` of type `float32`.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "FloatInput", name, a)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_float_input(
          (a, name,), None)
      if _result is not NotImplemented:
        return _result
      return float_input_eager_fallback(
          a, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            float_input, (), dict(a=a, name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_float_input(
        (a, name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "FloatInput", a=a, name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          float_input, (), dict(a=a, name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  return _op
FloatInput = tf_export("raw_ops.FloatInput")(_ops.to_raw_op(float_input))
_dispatcher_for_float_input = float_input._tf_type_based_dispatcher.Dispatch


def float_input_eager_fallback(a: Annotated[Any, _atypes.Float32], name, ctx):
  a = _ops.convert_to_tensor(a, _dtypes.float32)
  _inputs_flat = [a]
  _attrs = None
  _result = _execute.execute(b"FloatInput", 0, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  _result = None
  return _result


@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('float_output')
def float_output(name=None) -> Annotated[Any, _atypes.Float32]:
  r"""TODO: add doc.

  Args:
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `float32`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "FloatOutput", name)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_float_output(
          (name,), None)
      if _result is not NotImplemented:
        return _result
      return float_output_eager_fallback(
          name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            float_output, (), dict(name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_float_output(
        (name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "FloatOutput", name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          float_output, (), dict(name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ()
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "FloatOutput", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

FloatOutput = tf_export("raw_ops.FloatOutput")(_ops.to_raw_op(float_output))
_dispatcher_for_float_output = float_output._tf_type_based_dispatcher.Dispatch


def float_output_eager_fallback(name, ctx) -> Annotated[Any, _atypes.Float32]:
  _inputs_flat = []
  _attrs = None
  _result = _execute.execute(b"FloatOutput", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "FloatOutput", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

_FloatOutputStringOutputOutput = collections.namedtuple(
    "FloatOutputStringOutput",
    ["a", "b"])


@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('float_output_string_output')
def float_output_string_output(name=None):
  r"""TODO: add doc.

  Args:
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (a, b).

    a: A `Tensor` of type `float32`.
    b: A `Tensor` of type `string`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "FloatOutputStringOutput", name)
      _result = _FloatOutputStringOutputOutput._make(_result)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_float_output_string_output(
          (name,), None)
      if _result is not NotImplemented:
        return _result
      return float_output_string_output_eager_fallback(
          name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            float_output_string_output, (), dict(name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_float_output_string_output(
        (name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "FloatOutputStringOutput", name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          float_output_string_output, (), dict(name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ()
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "FloatOutputStringOutput", _inputs_flat, _attrs, _result)
  _result = _FloatOutputStringOutputOutput._make(_result)
  return _result

FloatOutputStringOutput = tf_export("raw_ops.FloatOutputStringOutput")(_ops.to_raw_op(float_output_string_output))
_dispatcher_for_float_output_string_output = float_output_string_output._tf_type_based_dispatcher.Dispatch


def float_output_string_output_eager_fallback(name, ctx):
  _inputs_flat = []
  _attrs = None
  _result = _execute.execute(b"FloatOutputStringOutput", 2,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "FloatOutputStringOutput", _inputs_flat, _attrs, _result)
  _result = _FloatOutputStringOutputOutput._make(_result)
  return _result

_Foo1Output = collections.namedtuple(
    "Foo1",
    ["d", "e"])


@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('foo1')
def foo1(a: Annotated[Any, _atypes.Float32], b: Annotated[Any, _atypes.Int32], c: Annotated[Any, _atypes.Int32], name=None):
  r"""TODO: add doc.

  Args:
    a: A `Tensor` of type `float32`.
    b: A `Tensor` of type `int32`.
    c: A `Tensor` of type `int32`.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (d, e).

    d: A `Tensor` of type `float32`.
    e: A `Tensor` of type `int32`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "Foo1", name, a, b, c)
      _result = _Foo1Output._make(_result)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_foo1(
          (a, b, c, name,), None)
      if _result is not NotImplemented:
        return _result
      return foo1_eager_fallback(
          a, b, c, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            foo1, (), dict(a=a, b=b, c=c, name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_foo1(
        (a, b, c, name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "Foo1", a=a, b=b, c=c, name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          foo1, (), dict(a=a, b=b, c=c, name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ()
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "Foo1", _inputs_flat, _attrs, _result)
  _result = _Foo1Output._make(_result)
  return _result

Foo1 = tf_export("raw_ops.Foo1")(_ops.to_raw_op(foo1))
_dispatcher_for_foo1 = foo1._tf_type_based_dispatcher.Dispatch


def foo1_eager_fallback(a: Annotated[Any, _atypes.Float32], b: Annotated[Any, _atypes.Int32], c: Annotated[Any, _atypes.Int32], name, ctx):
  a = _ops.convert_to_tensor(a, _dtypes.float32)
  b = _ops.convert_to_tensor(b, _dtypes.int32)
  c = _ops.convert_to_tensor(c, _dtypes.int32)
  _inputs_flat = [a, b, c]
  _attrs = None
  _result = _execute.execute(b"Foo1", 2, inputs=_inputs_flat, attrs=_attrs,
                             ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "Foo1", _inputs_flat, _attrs, _result)
  _result = _Foo1Output._make(_result)
  return _result

_Foo2Output = collections.namedtuple(
    "Foo2",
    ["d", "e"])


@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('foo2')
def foo2(a: Annotated[Any, _atypes.Float32], b: Annotated[Any, _atypes.String], c: Annotated[Any, _atypes.String], name=None):
  r"""TODO: add doc.

  Args:
    a: A `Tensor` of type `float32`.
    b: A `Tensor` of type `string`.
    c: A `Tensor` of type `string`.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (d, e).

    d: A `Tensor` of type `float32`.
    e: A `Tensor` of type `int32`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "Foo2", name, a, b, c)
      _result = _Foo2Output._make(_result)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_foo2(
          (a, b, c, name,), None)
      if _result is not NotImplemented:
        return _result
      return foo2_eager_fallback(
          a, b, c, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            foo2, (), dict(a=a, b=b, c=c, name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_foo2(
        (a, b, c, name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "Foo2", a=a, b=b, c=c, name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          foo2, (), dict(a=a, b=b, c=c, name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ()
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "Foo2", _inputs_flat, _attrs, _result)
  _result = _Foo2Output._make(_result)
  return _result

Foo2 = tf_export("raw_ops.Foo2")(_ops.to_raw_op(foo2))
_dispatcher_for_foo2 = foo2._tf_type_based_dispatcher.Dispatch


def foo2_eager_fallback(a: Annotated[Any, _atypes.Float32], b: Annotated[Any, _atypes.String], c: Annotated[Any, _atypes.String], name, ctx):
  a = _ops.convert_to_tensor(a, _dtypes.float32)
  b = _ops.convert_to_tensor(b, _dtypes.string)
  c = _ops.convert_to_tensor(c, _dtypes.string)
  _inputs_flat = [a, b, c]
  _attrs = None
  _result = _execute.execute(b"Foo2", 2, inputs=_inputs_flat, attrs=_attrs,
                             ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "Foo2", _inputs_flat, _attrs, _result)
  _result = _Foo2Output._make(_result)
  return _result

_Foo3Output = collections.namedtuple(
    "Foo3",
    ["d", "e"])


@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('foo3')
def foo3(a: Annotated[Any, _atypes.Float32], b: Annotated[Any, _atypes.String], c: Annotated[Any, _atypes.Float32], name=None):
  r"""TODO: add doc.

  Args:
    a: A `Tensor` of type `float32`.
    b: A `Tensor` of type `string`.
    c: A `Tensor` of type `float32`.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (d, e).

    d: A `Tensor` of type `float32`.
    e: A `Tensor` of type `int32`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "Foo3", name, a, b, c)
      _result = _Foo3Output._make(_result)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_foo3(
          (a, b, c, name,), None)
      if _result is not NotImplemented:
        return _result
      return foo3_eager_fallback(
          a, b, c, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            foo3, (), dict(a=a, b=b, c=c, name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_foo3(
        (a, b, c, name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "Foo3", a=a, b=b, c=c, name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          foo3, (), dict(a=a, b=b, c=c, name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ()
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "Foo3", _inputs_flat, _attrs, _result)
  _result = _Foo3Output._make(_result)
  return _result

Foo3 = tf_export("raw_ops.Foo3")(_ops.to_raw_op(foo3))
_dispatcher_for_foo3 = foo3._tf_type_based_dispatcher.Dispatch


def foo3_eager_fallback(a: Annotated[Any, _atypes.Float32], b: Annotated[Any, _atypes.String], c: Annotated[Any, _atypes.Float32], name, ctx):
  a = _ops.convert_to_tensor(a, _dtypes.float32)
  b = _ops.convert_to_tensor(b, _dtypes.string)
  c = _ops.convert_to_tensor(c, _dtypes.float32)
  _inputs_flat = [a, b, c]
  _attrs = None
  _result = _execute.execute(b"Foo3", 2, inputs=_inputs_flat, attrs=_attrs,
                             ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "Foo3", _inputs_flat, _attrs, _result)
  _result = _Foo3Output._make(_result)
  return _result


@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('func_attr')
def func_attr(f, name=None):
  r"""TODO: add doc.

  Args:
    f: A function decorated with @Defun.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "FuncAttr", name, "f", f)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_func_attr(
          (f, name,), None)
      if _result is not NotImplemented:
        return _result
      return func_attr_eager_fallback(
          f=f, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            func_attr, (), dict(f=f, name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_func_attr(
        (f, name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "FuncAttr", f=f, name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          func_attr, (), dict(f=f, name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  return _op
FuncAttr = tf_export("raw_ops.FuncAttr")(_ops.to_raw_op(func_attr))
_dispatcher_for_func_attr = func_attr._tf_type_based_dispatcher.Dispatch


def func_attr_eager_fallback(f, name, ctx):
  _inputs_flat = []
  _attrs = ("f", f)
  _result = _execute.execute(b"FuncAttr", 0, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  _result = None
  return _result


@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('func_list_attr')
def func_list_attr(f, name=None):
  r"""TODO: add doc.

  Args:
    f: A list of functions decorated with @Defun.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "FuncListAttr", name, "f", f)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_func_list_attr(
          (f, name,), None)
      if _result is not NotImplemented:
        return _result
      return func_list_attr_eager_fallback(
          f=f, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            func_list_attr, (), dict(f=f, name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_func_list_attr(
        (f, name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  if not isinstance(f, (list, tuple)):
    raise TypeError(
        "Expected list for 'f' argument to "
        "'func_list_attr' Op, not %r." % f)
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "FuncListAttr", f=f, name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          func_list_attr, (), dict(f=f, name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  return _op
FuncListAttr = tf_export("raw_ops.FuncListAttr")(_ops.to_raw_op(func_list_attr))
_dispatcher_for_func_list_attr = func_list_attr._tf_type_based_dispatcher.Dispatch


def func_list_attr_eager_fallback(f, name, ctx):
  if not isinstance(f, (list, tuple)):
    raise TypeError(
        "Expected list for 'f' argument to "
        "'func_list_attr' Op, not %r." % f)
  _inputs_flat = []
  _attrs = ("f", f)
  _result = _execute.execute(b"FuncListAttr", 0, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  _result = None
  return _result


@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('get_deadline')
def get_deadline(name=None) -> Annotated[Any, _atypes.Int64]:
  r"""TODO: add doc.

  Args:
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `int64`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "GetDeadline", name)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_get_deadline(
          (name,), None)
      if _result is not NotImplemented:
        return _result
      return get_deadline_eager_fallback(
          name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            get_deadline, (), dict(name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_get_deadline(
        (name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "GetDeadline", name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          get_deadline, (), dict(name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ()
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "GetDeadline", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

GetDeadline = tf_export("raw_ops.GetDeadline")(_ops.to_raw_op(get_deadline))
_dispatcher_for_get_deadline = get_deadline._tf_type_based_dispatcher.Dispatch


def get_deadline_eager_fallback(name, ctx) -> Annotated[Any, _atypes.Int64]:
  _inputs_flat = []
  _attrs = None
  _result = _execute.execute(b"GetDeadline", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "GetDeadline", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('graph_def_version')
def graph_def_version(name=None) -> Annotated[Any, _atypes.Int32]:
  r"""TODO: add doc.

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
        _ctx, "GraphDefVersion", name)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_graph_def_version(
          (name,), None)
      if _result is not NotImplemented:
        return _result
      return graph_def_version_eager_fallback(
          name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            graph_def_version, (), dict(name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_graph_def_version(
        (name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "GraphDefVersion", name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          graph_def_version, (), dict(name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ()
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "GraphDefVersion", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

GraphDefVersion = tf_export("raw_ops.GraphDefVersion")(_ops.to_raw_op(graph_def_version))
_dispatcher_for_graph_def_version = graph_def_version._tf_type_based_dispatcher.Dispatch


def graph_def_version_eager_fallback(name, ctx) -> Annotated[Any, _atypes.Int32]:
  _inputs_flat = []
  _attrs = None
  _result = _execute.execute(b"GraphDefVersion", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "GraphDefVersion", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_InPolymorphicTwice_T = TypeVar("TV_InPolymorphicTwice_T", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float16, _atypes.Float32, _atypes.Float64, _atypes.Float8e4m3fn, _atypes.Float8e5m2, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int4, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.Resource, _atypes.String, _atypes.UInt16, _atypes.UInt32, _atypes.UInt4, _atypes.UInt64, _atypes.UInt8, _atypes.Variant)

@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('in_polymorphic_twice')
def in_polymorphic_twice(a: Annotated[List[Any], TV_InPolymorphicTwice_T], b: Annotated[List[Any], TV_InPolymorphicTwice_T], name=None):
  r"""TODO: add doc.

  Args:
    a: A list of `Tensor` objects with the same type.
    b: A list of `Tensor` objects with the same type as `a`.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "InPolymorphicTwice", name, a, b)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_in_polymorphic_twice(
          (a, b, name,), None)
      if _result is not NotImplemented:
        return _result
      return in_polymorphic_twice_eager_fallback(
          a, b, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            in_polymorphic_twice, (), dict(a=a, b=b, name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_in_polymorphic_twice(
        (a, b, name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  if not isinstance(a, (list, tuple)):
    raise TypeError(
        "Expected list for 'a' argument to "
        "'in_polymorphic_twice' Op, not %r." % a)
  _attr_N = len(a)
  if not isinstance(b, (list, tuple)):
    raise TypeError(
        "Expected list for 'b' argument to "
        "'in_polymorphic_twice' Op, not %r." % b)
  _attr_M = len(b)
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "InPolymorphicTwice", a=a, b=b, name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          in_polymorphic_twice, (), dict(a=a, b=b, name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  return _op
InPolymorphicTwice = tf_export("raw_ops.InPolymorphicTwice")(_ops.to_raw_op(in_polymorphic_twice))
_dispatcher_for_in_polymorphic_twice = in_polymorphic_twice._tf_type_based_dispatcher.Dispatch


def in_polymorphic_twice_eager_fallback(a: Annotated[List[Any], TV_InPolymorphicTwice_T], b: Annotated[List[Any], TV_InPolymorphicTwice_T], name, ctx):
  if not isinstance(a, (list, tuple)):
    raise TypeError(
        "Expected list for 'a' argument to "
        "'in_polymorphic_twice' Op, not %r." % a)
  _attr_N = len(a)
  if not isinstance(b, (list, tuple)):
    raise TypeError(
        "Expected list for 'b' argument to "
        "'in_polymorphic_twice' Op, not %r." % b)
  _attr_M = len(b)
  _attr_T, _inputs_T = _execute.args_to_matching_eager(list(a) + list(b), ctx, [], _dtypes.int32)
  _inputs_T = [_inputs_T[:_attr_N]] + _inputs_T[_attr_N:]
  _inputs_T = _inputs_T[:1] + [_inputs_T[1:]]
  (a, b) = _inputs_T
  _inputs_flat = list(a) + list(b)
  _attrs = ("T", _attr_T, "N", _attr_N, "M", _attr_M)
  _result = _execute.execute(b"InPolymorphicTwice", 0, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  _result = None
  return _result


@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('int64_output')
def int64_output(name=None) -> Annotated[Any, _atypes.Int64]:
  r"""TODO: add doc.

  Args:
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `int64`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "Int64Output", name)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_int64_output(
          (name,), None)
      if _result is not NotImplemented:
        return _result
      return int64_output_eager_fallback(
          name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            int64_output, (), dict(name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_int64_output(
        (name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "Int64Output", name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          int64_output, (), dict(name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ()
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "Int64Output", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

Int64Output = tf_export("raw_ops.Int64Output")(_ops.to_raw_op(int64_output))
_dispatcher_for_int64_output = int64_output._tf_type_based_dispatcher.Dispatch


def int64_output_eager_fallback(name, ctx) -> Annotated[Any, _atypes.Int64]:
  _inputs_flat = []
  _attrs = None
  _result = _execute.execute(b"Int64Output", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "Int64Output", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('int_attr')
def int_attr(foo:int=1, name=None) -> Annotated[Any, _atypes.Int64]:
  r"""TODO: add doc.

  Args:
    foo: An optional `int`. Defaults to `1`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `int64`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "IntAttr", name, "foo", foo)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_int_attr(
          (foo, name,), None)
      if _result is not NotImplemented:
        return _result
      return int_attr_eager_fallback(
          foo=foo, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            int_attr, (), dict(foo=foo, name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_int_attr(
        (foo, name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  if foo is None:
    foo = 1
  foo = _execute.make_int(foo, "foo")
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "IntAttr", foo=foo, name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          int_attr, (), dict(foo=foo, name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("foo", _op._get_attr_int("foo"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "IntAttr", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

IntAttr = tf_export("raw_ops.IntAttr")(_ops.to_raw_op(int_attr))
_dispatcher_for_int_attr = int_attr._tf_type_based_dispatcher.Dispatch


def int_attr_eager_fallback(foo: int, name, ctx) -> Annotated[Any, _atypes.Int64]:
  if foo is None:
    foo = 1
  foo = _execute.make_int(foo, "foo")
  _inputs_flat = []
  _attrs = ("foo", foo)
  _result = _execute.execute(b"IntAttr", 1, inputs=_inputs_flat, attrs=_attrs,
                             ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "IntAttr", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('int_input')
def int_input(a: Annotated[Any, _atypes.Int32], name=None):
  r"""TODO: add doc.

  Args:
    a: A `Tensor` of type `int32`.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "IntInput", name, a)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_int_input(
          (a, name,), None)
      if _result is not NotImplemented:
        return _result
      return int_input_eager_fallback(
          a, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            int_input, (), dict(a=a, name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_int_input(
        (a, name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "IntInput", a=a, name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          int_input, (), dict(a=a, name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  return _op
IntInput = tf_export("raw_ops.IntInput")(_ops.to_raw_op(int_input))
_dispatcher_for_int_input = int_input._tf_type_based_dispatcher.Dispatch


def int_input_eager_fallback(a: Annotated[Any, _atypes.Int32], name, ctx):
  a = _ops.convert_to_tensor(a, _dtypes.int32)
  _inputs_flat = [a]
  _attrs = None
  _result = _execute.execute(b"IntInput", 0, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  _result = None
  return _result


@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('int_input_float_input')
def int_input_float_input(a: Annotated[Any, _atypes.Int32], b: Annotated[Any, _atypes.Float32], name=None):
  r"""TODO: add doc.

  Args:
    a: A `Tensor` of type `int32`.
    b: A `Tensor` of type `float32`.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "IntInputFloatInput", name, a, b)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_int_input_float_input(
          (a, b, name,), None)
      if _result is not NotImplemented:
        return _result
      return int_input_float_input_eager_fallback(
          a, b, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            int_input_float_input, (), dict(a=a, b=b, name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_int_input_float_input(
        (a, b, name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "IntInputFloatInput", a=a, b=b, name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          int_input_float_input, (), dict(a=a, b=b, name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  return _op
IntInputFloatInput = tf_export("raw_ops.IntInputFloatInput")(_ops.to_raw_op(int_input_float_input))
_dispatcher_for_int_input_float_input = int_input_float_input._tf_type_based_dispatcher.Dispatch


def int_input_float_input_eager_fallback(a: Annotated[Any, _atypes.Int32], b: Annotated[Any, _atypes.Float32], name, ctx):
  a = _ops.convert_to_tensor(a, _dtypes.int32)
  b = _ops.convert_to_tensor(b, _dtypes.float32)
  _inputs_flat = [a, b]
  _attrs = None
  _result = _execute.execute(b"IntInputFloatInput", 0, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  _result = None
  return _result


@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('int_input_int_output')
def int_input_int_output(a: Annotated[Any, _atypes.Int32], name=None) -> Annotated[Any, _atypes.Int32]:
  r"""TODO: add doc.

  Args:
    a: A `Tensor` of type `int32`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `int32`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "IntInputIntOutput", name, a)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_int_input_int_output(
          (a, name,), None)
      if _result is not NotImplemented:
        return _result
      return int_input_int_output_eager_fallback(
          a, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            int_input_int_output, (), dict(a=a, name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_int_input_int_output(
        (a, name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "IntInputIntOutput", a=a, name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          int_input_int_output, (), dict(a=a, name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ()
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "IntInputIntOutput", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

IntInputIntOutput = tf_export("raw_ops.IntInputIntOutput")(_ops.to_raw_op(int_input_int_output))
_dispatcher_for_int_input_int_output = int_input_int_output._tf_type_based_dispatcher.Dispatch


def int_input_int_output_eager_fallback(a: Annotated[Any, _atypes.Int32], name, ctx) -> Annotated[Any, _atypes.Int32]:
  a = _ops.convert_to_tensor(a, _dtypes.int32)
  _inputs_flat = [a]
  _attrs = None
  _result = _execute.execute(b"IntInputIntOutput", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "IntInputIntOutput", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('int_output')
def int_output(name=None) -> Annotated[Any, _atypes.Int32]:
  r"""TODO: add doc.

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
        _ctx, "IntOutput", name)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_int_output(
          (name,), None)
      if _result is not NotImplemented:
        return _result
      return int_output_eager_fallback(
          name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            int_output, (), dict(name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_int_output(
        (name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "IntOutput", name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          int_output, (), dict(name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ()
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "IntOutput", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

IntOutput = tf_export("raw_ops.IntOutput")(_ops.to_raw_op(int_output))
_dispatcher_for_int_output = int_output._tf_type_based_dispatcher.Dispatch


def int_output_eager_fallback(name, ctx) -> Annotated[Any, _atypes.Int32]:
  _inputs_flat = []
  _attrs = None
  _result = _execute.execute(b"IntOutput", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "IntOutput", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

_IntOutputFloatOutputOutput = collections.namedtuple(
    "IntOutputFloatOutput",
    ["a", "b"])


@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('int_output_float_output')
def int_output_float_output(name=None):
  r"""TODO: add doc.

  Args:
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (a, b).

    a: A `Tensor` of type `int32`.
    b: A `Tensor` of type `float32`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "IntOutputFloatOutput", name)
      _result = _IntOutputFloatOutputOutput._make(_result)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_int_output_float_output(
          (name,), None)
      if _result is not NotImplemented:
        return _result
      return int_output_float_output_eager_fallback(
          name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            int_output_float_output, (), dict(name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_int_output_float_output(
        (name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "IntOutputFloatOutput", name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          int_output_float_output, (), dict(name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ()
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "IntOutputFloatOutput", _inputs_flat, _attrs, _result)
  _result = _IntOutputFloatOutputOutput._make(_result)
  return _result

IntOutputFloatOutput = tf_export("raw_ops.IntOutputFloatOutput")(_ops.to_raw_op(int_output_float_output))
_dispatcher_for_int_output_float_output = int_output_float_output._tf_type_based_dispatcher.Dispatch


def int_output_float_output_eager_fallback(name, ctx):
  _inputs_flat = []
  _attrs = None
  _result = _execute.execute(b"IntOutputFloatOutput", 2, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "IntOutputFloatOutput", _inputs_flat, _attrs, _result)
  _result = _IntOutputFloatOutputOutput._make(_result)
  return _result


@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('is_resource_handle_ref_counting')
def is_resource_handle_ref_counting(handle: Annotated[Any, _atypes.Resource], name=None) -> Annotated[Any, _atypes.Bool]:
  r"""TODO: add doc.

  Args:
    handle: A `Tensor` of type `resource`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `bool`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "IsResourceHandleRefCounting", name, handle)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_is_resource_handle_ref_counting(
          (handle, name,), None)
      if _result is not NotImplemented:
        return _result
      return is_resource_handle_ref_counting_eager_fallback(
          handle, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            is_resource_handle_ref_counting, (), dict(handle=handle,
                                                      name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_is_resource_handle_ref_counting(
        (handle, name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "IsResourceHandleRefCounting", handle=handle, name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          is_resource_handle_ref_counting, (), dict(handle=handle, name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ()
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "IsResourceHandleRefCounting", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

IsResourceHandleRefCounting = tf_export("raw_ops.IsResourceHandleRefCounting")(_ops.to_raw_op(is_resource_handle_ref_counting))
_dispatcher_for_is_resource_handle_ref_counting = is_resource_handle_ref_counting._tf_type_based_dispatcher.Dispatch


def is_resource_handle_ref_counting_eager_fallback(handle: Annotated[Any, _atypes.Resource], name, ctx) -> Annotated[Any, _atypes.Bool]:
  handle = _ops.convert_to_tensor(handle, _dtypes.resource)
  _inputs_flat = [handle]
  _attrs = None
  _result = _execute.execute(b"IsResourceHandleRefCounting", 1,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "IsResourceHandleRefCounting", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('is_tensor_float32_enabled')
def is_tensor_float32_enabled(name=None) -> Annotated[Any, _atypes.Bool]:
  r"""TODO: add doc.

  Args:
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `bool`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "IsTensorFloat32Enabled", name)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_is_tensor_float32_enabled(
          (name,), None)
      if _result is not NotImplemented:
        return _result
      return is_tensor_float32_enabled_eager_fallback(
          name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            is_tensor_float32_enabled, (), dict(name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_is_tensor_float32_enabled(
        (name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "IsTensorFloat32Enabled", name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          is_tensor_float32_enabled, (), dict(name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ()
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "IsTensorFloat32Enabled", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

IsTensorFloat32Enabled = tf_export("raw_ops.IsTensorFloat32Enabled")(_ops.to_raw_op(is_tensor_float32_enabled))
_dispatcher_for_is_tensor_float32_enabled = is_tensor_float32_enabled._tf_type_based_dispatcher.Dispatch


def is_tensor_float32_enabled_eager_fallback(name, ctx) -> Annotated[Any, _atypes.Bool]:
  _inputs_flat = []
  _attrs = None
  _result = _execute.execute(b"IsTensorFloat32Enabled", 1,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "IsTensorFloat32Enabled", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('kernel_label')
def kernel_label(name=None) -> Annotated[Any, _atypes.String]:
  r"""TODO: add doc.

  Args:
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `string`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "KernelLabel", name)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_kernel_label(
          (name,), None)
      if _result is not NotImplemented:
        return _result
      return kernel_label_eager_fallback(
          name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            kernel_label, (), dict(name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_kernel_label(
        (name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "KernelLabel", name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          kernel_label, (), dict(name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ()
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "KernelLabel", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

KernelLabel = tf_export("raw_ops.KernelLabel")(_ops.to_raw_op(kernel_label))
_dispatcher_for_kernel_label = kernel_label._tf_type_based_dispatcher.Dispatch


def kernel_label_eager_fallback(name, ctx) -> Annotated[Any, _atypes.String]:
  _inputs_flat = []
  _attrs = None
  _result = _execute.execute(b"KernelLabel", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "KernelLabel", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('kernel_label_required')
def kernel_label_required(input: Annotated[Any, _atypes.Int32], name=None) -> Annotated[Any, _atypes.String]:
  r"""TODO: add doc.

  Args:
    input: A `Tensor` of type `int32`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `string`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "KernelLabelRequired", name, input)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_kernel_label_required(
          (input, name,), None)
      if _result is not NotImplemented:
        return _result
      return kernel_label_required_eager_fallback(
          input, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            kernel_label_required, (), dict(input=input, name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_kernel_label_required(
        (input, name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "KernelLabelRequired", input=input, name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          kernel_label_required, (), dict(input=input, name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ()
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "KernelLabelRequired", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

KernelLabelRequired = tf_export("raw_ops.KernelLabelRequired")(_ops.to_raw_op(kernel_label_required))
_dispatcher_for_kernel_label_required = kernel_label_required._tf_type_based_dispatcher.Dispatch


def kernel_label_required_eager_fallback(input: Annotated[Any, _atypes.Int32], name, ctx) -> Annotated[Any, _atypes.String]:
  input = _ops.convert_to_tensor(input, _dtypes.int32)
  _inputs_flat = [input]
  _attrs = None
  _result = _execute.execute(b"KernelLabelRequired", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "KernelLabelRequired", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_ListInput_T = TypeVar("TV_ListInput_T", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float16, _atypes.Float32, _atypes.Float64, _atypes.Float8e4m3fn, _atypes.Float8e5m2, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int4, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.Resource, _atypes.String, _atypes.UInt16, _atypes.UInt32, _atypes.UInt4, _atypes.UInt64, _atypes.UInt8, _atypes.Variant)

@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('list_input')
def list_input(a: Annotated[List[Any], TV_ListInput_T], name=None):
  r"""TODO: add doc.

  Args:
    a: A list of at least 1 `Tensor` objects with the same type.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "ListInput", name, a)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_list_input(
          (a, name,), None)
      if _result is not NotImplemented:
        return _result
      return list_input_eager_fallback(
          a, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            list_input, (), dict(a=a, name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_list_input(
        (a, name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  if not isinstance(a, (list, tuple)):
    raise TypeError(
        "Expected list for 'a' argument to "
        "'list_input' Op, not %r." % a)
  _attr_N = len(a)
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "ListInput", a=a, name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          list_input, (), dict(a=a, name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  return _op
ListInput = tf_export("raw_ops.ListInput")(_ops.to_raw_op(list_input))
_dispatcher_for_list_input = list_input._tf_type_based_dispatcher.Dispatch


def list_input_eager_fallback(a: Annotated[List[Any], TV_ListInput_T], name, ctx):
  if not isinstance(a, (list, tuple)):
    raise TypeError(
        "Expected list for 'a' argument to "
        "'list_input' Op, not %r." % a)
  _attr_N = len(a)
  _attr_T, a = _execute.args_to_matching_eager(list(a), ctx, [])
  _inputs_flat = list(a)
  _attrs = ("N", _attr_N, "T", _attr_T)
  _result = _execute.execute(b"ListInput", 0, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  _result = None
  return _result


@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('list_output')
def list_output(T, name=None):
  r"""TODO: add doc.

  Args:
    T: A list of `tf.DTypes` that has length `>= 1`.
    name: A name for the operation (optional).

  Returns:
    A list of `Tensor` objects of type `T`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "ListOutput", name, "T", T)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_list_output(
          (T, name,), None)
      if _result is not NotImplemented:
        return _result
      return list_output_eager_fallback(
          T=T, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            list_output, (), dict(T=T, name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_list_output(
        (T, name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  if not isinstance(T, (list, tuple)):
    raise TypeError(
        "Expected list for 'T' argument to "
        "'list_output' Op, not %r." % T)
  T = [_execute.make_type(_t, "T") for _t in T]
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "ListOutput", T=T, name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          list_output, (), dict(T=T, name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op.get_attr("T"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "ListOutput", _inputs_flat, _attrs, _result)
  return _result

ListOutput = tf_export("raw_ops.ListOutput")(_ops.to_raw_op(list_output))
_dispatcher_for_list_output = list_output._tf_type_based_dispatcher.Dispatch


def list_output_eager_fallback(T, name, ctx):
  if not isinstance(T, (list, tuple)):
    raise TypeError(
        "Expected list for 'T' argument to "
        "'list_output' Op, not %r." % T)
  T = [_execute.make_type(_t, "T") for _t in T]
  _inputs_flat = []
  _attrs = ("T", T)
  _result = _execute.execute(b"ListOutput", len(T), inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "ListOutput", _inputs_flat, _attrs, _result)
  return _result


@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('make_weak_resource_handle')
def make_weak_resource_handle(handle: Annotated[Any, _atypes.Resource], name=None) -> Annotated[Any, _atypes.Resource]:
  r"""TODO: add doc.

  Args:
    handle: A `Tensor` of type `resource`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `resource`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "MakeWeakResourceHandle", name, handle)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_make_weak_resource_handle(
          (handle, name,), None)
      if _result is not NotImplemented:
        return _result
      return make_weak_resource_handle_eager_fallback(
          handle, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            make_weak_resource_handle, (), dict(handle=handle, name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_make_weak_resource_handle(
        (handle, name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "MakeWeakResourceHandle", handle=handle, name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          make_weak_resource_handle, (), dict(handle=handle, name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ()
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "MakeWeakResourceHandle", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

MakeWeakResourceHandle = tf_export("raw_ops.MakeWeakResourceHandle")(_ops.to_raw_op(make_weak_resource_handle))
_dispatcher_for_make_weak_resource_handle = make_weak_resource_handle._tf_type_based_dispatcher.Dispatch


def make_weak_resource_handle_eager_fallback(handle: Annotated[Any, _atypes.Resource], name, ctx) -> Annotated[Any, _atypes.Resource]:
  handle = _ops.convert_to_tensor(handle, _dtypes.resource)
  _inputs_flat = [handle]
  _attrs = None
  _result = _execute.execute(b"MakeWeakResourceHandle", 1,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "MakeWeakResourceHandle", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

_MixedStructOutput = collections.namedtuple(
    "MixedStruct",
    ["a", "b"])


@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('mixed_struct')
def mixed_struct(n_a: int, name=None):
  r"""TODO: add doc.

  Args:
    n_a: An `int` that is `>= 0`.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (a, b).

    a: A list of `n_a` `Tensor` objects with type `int32`.
    b: A `Tensor` of type `float32`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "MixedStruct", name, "n_a", n_a)
      _result = _MixedStructOutput._make(_result)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_mixed_struct(
          (n_a, name,), None)
      if _result is not NotImplemented:
        return _result
      return mixed_struct_eager_fallback(
          n_a=n_a, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            mixed_struct, (), dict(n_a=n_a, name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_mixed_struct(
        (n_a, name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  n_a = _execute.make_int(n_a, "n_a")
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "MixedStruct", n_a=n_a, name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          mixed_struct, (), dict(n_a=n_a, name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("n_a", _op._get_attr_int("n_a"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "MixedStruct", _inputs_flat, _attrs, _result)
  _result = [_result[:n_a]] + _result[n_a:]
  _result = _MixedStructOutput._make(_result)
  return _result

MixedStruct = tf_export("raw_ops.MixedStruct")(_ops.to_raw_op(mixed_struct))
_dispatcher_for_mixed_struct = mixed_struct._tf_type_based_dispatcher.Dispatch


def mixed_struct_eager_fallback(n_a: int, name, ctx):
  n_a = _execute.make_int(n_a, "n_a")
  _inputs_flat = []
  _attrs = ("n_a", n_a)
  _result = _execute.execute(b"MixedStruct", n_a + 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "MixedStruct", _inputs_flat, _attrs, _result)
  _result = [_result[:n_a]] + _result[n_a:]
  _result = _MixedStructOutput._make(_result)
  return _result


TV_NInPolymorphicTwice_T = TypeVar("TV_NInPolymorphicTwice_T", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float16, _atypes.Float32, _atypes.Float64, _atypes.Float8e4m3fn, _atypes.Float8e5m2, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int4, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.Resource, _atypes.String, _atypes.UInt16, _atypes.UInt32, _atypes.UInt4, _atypes.UInt64, _atypes.UInt8, _atypes.Variant)

@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('n_in_polymorphic_twice')
def n_in_polymorphic_twice(a: Annotated[List[Any], TV_NInPolymorphicTwice_T], b: Annotated[List[Any], TV_NInPolymorphicTwice_T], name=None):
  r"""TODO: add doc.

  Args:
    a: A list of `Tensor` objects with the same type.
    b: A list with the same length as `a` of `Tensor` objects with the same type as `a`.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "NInPolymorphicTwice", name, a, b)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_n_in_polymorphic_twice(
          (a, b, name,), None)
      if _result is not NotImplemented:
        return _result
      return n_in_polymorphic_twice_eager_fallback(
          a, b, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            n_in_polymorphic_twice, (), dict(a=a, b=b, name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_n_in_polymorphic_twice(
        (a, b, name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  if not isinstance(a, (list, tuple)):
    raise TypeError(
        "Expected list for 'a' argument to "
        "'n_in_polymorphic_twice' Op, not %r." % a)
  _attr_N = len(a)
  if not isinstance(b, (list, tuple)):
    raise TypeError(
        "Expected list for 'b' argument to "
        "'n_in_polymorphic_twice' Op, not %r." % b)
  if len(b) != _attr_N:
    raise ValueError(
        "List argument 'b' to 'n_in_polymorphic_twice' Op with length %d "
        "must match length %d of argument 'a'." %
        (len(b), _attr_N))
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "NInPolymorphicTwice", a=a, b=b, name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          n_in_polymorphic_twice, (), dict(a=a, b=b, name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  return _op
NInPolymorphicTwice = tf_export("raw_ops.NInPolymorphicTwice")(_ops.to_raw_op(n_in_polymorphic_twice))
_dispatcher_for_n_in_polymorphic_twice = n_in_polymorphic_twice._tf_type_based_dispatcher.Dispatch


def n_in_polymorphic_twice_eager_fallback(a: Annotated[List[Any], TV_NInPolymorphicTwice_T], b: Annotated[List[Any], TV_NInPolymorphicTwice_T], name, ctx):
  if not isinstance(a, (list, tuple)):
    raise TypeError(
        "Expected list for 'a' argument to "
        "'n_in_polymorphic_twice' Op, not %r." % a)
  _attr_N = len(a)
  if not isinstance(b, (list, tuple)):
    raise TypeError(
        "Expected list for 'b' argument to "
        "'n_in_polymorphic_twice' Op, not %r." % b)
  if len(b) != _attr_N:
    raise ValueError(
        "List argument 'b' to 'n_in_polymorphic_twice' Op with length %d "
        "must match length %d of argument 'a'." %
        (len(b), _attr_N))
  _attr_T, _inputs_T = _execute.args_to_matching_eager(list(a) + list(b), ctx, [])
  _inputs_T = [_inputs_T[:_attr_N]] + _inputs_T[_attr_N:]
  _inputs_T = _inputs_T[:1] + [_inputs_T[1:]]
  (a, b) = _inputs_T
  _inputs_flat = list(a) + list(b)
  _attrs = ("T", _attr_T, "N", _attr_N)
  _result = _execute.execute(b"NInPolymorphicTwice", 0, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  _result = None
  return _result


@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('n_in_twice')
def n_in_twice(a: Annotated[List[Any], _atypes.Int32], b: Annotated[List[Any], _atypes.String], name=None):
  r"""TODO: add doc.

  Args:
    a: A list of `Tensor` objects with type `int32`.
    b: A list with the same length as `a` of `Tensor` objects with type `string`.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "NInTwice", name, a, b)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_n_in_twice(
          (a, b, name,), None)
      if _result is not NotImplemented:
        return _result
      return n_in_twice_eager_fallback(
          a, b, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            n_in_twice, (), dict(a=a, b=b, name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_n_in_twice(
        (a, b, name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  if not isinstance(a, (list, tuple)):
    raise TypeError(
        "Expected list for 'a' argument to "
        "'n_in_twice' Op, not %r." % a)
  _attr_N = len(a)
  if not isinstance(b, (list, tuple)):
    raise TypeError(
        "Expected list for 'b' argument to "
        "'n_in_twice' Op, not %r." % b)
  if len(b) != _attr_N:
    raise ValueError(
        "List argument 'b' to 'n_in_twice' Op with length %d "
        "must match length %d of argument 'a'." %
        (len(b), _attr_N))
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "NInTwice", a=a, b=b, name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          n_in_twice, (), dict(a=a, b=b, name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  return _op
NInTwice = tf_export("raw_ops.NInTwice")(_ops.to_raw_op(n_in_twice))
_dispatcher_for_n_in_twice = n_in_twice._tf_type_based_dispatcher.Dispatch


def n_in_twice_eager_fallback(a: Annotated[List[Any], _atypes.Int32], b: Annotated[List[Any], _atypes.String], name, ctx):
  if not isinstance(a, (list, tuple)):
    raise TypeError(
        "Expected list for 'a' argument to "
        "'n_in_twice' Op, not %r." % a)
  _attr_N = len(a)
  if not isinstance(b, (list, tuple)):
    raise TypeError(
        "Expected list for 'b' argument to "
        "'n_in_twice' Op, not %r." % b)
  if len(b) != _attr_N:
    raise ValueError(
        "List argument 'b' to 'n_in_twice' Op with length %d "
        "must match length %d of argument 'a'." %
        (len(b), _attr_N))
  a = _ops.convert_n_to_tensor(a, _dtypes.int32)
  b = _ops.convert_n_to_tensor(b, _dtypes.string)
  _inputs_flat = list(a) + list(b)
  _attrs = ("N", _attr_N)
  _result = _execute.execute(b"NInTwice", 0, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  _result = None
  return _result


TV_NInTwoTypeVariables_S = TypeVar("TV_NInTwoTypeVariables_S", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float16, _atypes.Float32, _atypes.Float64, _atypes.Float8e4m3fn, _atypes.Float8e5m2, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int4, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.Resource, _atypes.String, _atypes.UInt16, _atypes.UInt32, _atypes.UInt4, _atypes.UInt64, _atypes.UInt8, _atypes.Variant)
TV_NInTwoTypeVariables_T = TypeVar("TV_NInTwoTypeVariables_T", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float16, _atypes.Float32, _atypes.Float64, _atypes.Float8e4m3fn, _atypes.Float8e5m2, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int4, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.Resource, _atypes.String, _atypes.UInt16, _atypes.UInt32, _atypes.UInt4, _atypes.UInt64, _atypes.UInt8, _atypes.Variant)

@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('n_in_two_type_variables')
def n_in_two_type_variables(a: Annotated[List[Any], TV_NInTwoTypeVariables_S], b: Annotated[List[Any], TV_NInTwoTypeVariables_T], name=None):
  r"""TODO: add doc.

  Args:
    a: A list of `Tensor` objects with the same type.
    b: A list with the same length as `a` of `Tensor` objects with the same type.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "NInTwoTypeVariables", name, a, b)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_n_in_two_type_variables(
          (a, b, name,), None)
      if _result is not NotImplemented:
        return _result
      return n_in_two_type_variables_eager_fallback(
          a, b, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            n_in_two_type_variables, (), dict(a=a, b=b, name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_n_in_two_type_variables(
        (a, b, name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  if not isinstance(a, (list, tuple)):
    raise TypeError(
        "Expected list for 'a' argument to "
        "'n_in_two_type_variables' Op, not %r." % a)
  _attr_N = len(a)
  if not isinstance(b, (list, tuple)):
    raise TypeError(
        "Expected list for 'b' argument to "
        "'n_in_two_type_variables' Op, not %r." % b)
  if len(b) != _attr_N:
    raise ValueError(
        "List argument 'b' to 'n_in_two_type_variables' Op with length %d "
        "must match length %d of argument 'a'." %
        (len(b), _attr_N))
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "NInTwoTypeVariables", a=a, b=b, name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          n_in_two_type_variables, (), dict(a=a, b=b, name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  return _op
NInTwoTypeVariables = tf_export("raw_ops.NInTwoTypeVariables")(_ops.to_raw_op(n_in_two_type_variables))
_dispatcher_for_n_in_two_type_variables = n_in_two_type_variables._tf_type_based_dispatcher.Dispatch


def n_in_two_type_variables_eager_fallback(a: Annotated[List[Any], TV_NInTwoTypeVariables_S], b: Annotated[List[Any], TV_NInTwoTypeVariables_T], name, ctx):
  if not isinstance(a, (list, tuple)):
    raise TypeError(
        "Expected list for 'a' argument to "
        "'n_in_two_type_variables' Op, not %r." % a)
  _attr_N = len(a)
  if not isinstance(b, (list, tuple)):
    raise TypeError(
        "Expected list for 'b' argument to "
        "'n_in_two_type_variables' Op, not %r." % b)
  if len(b) != _attr_N:
    raise ValueError(
        "List argument 'b' to 'n_in_two_type_variables' Op with length %d "
        "must match length %d of argument 'a'." %
        (len(b), _attr_N))
  _attr_S, a = _execute.args_to_matching_eager(list(a), ctx, [])
  _attr_T, b = _execute.args_to_matching_eager(list(b), ctx, [])
  _inputs_flat = list(a) + list(b)
  _attrs = ("S", _attr_S, "T", _attr_T, "N", _attr_N)
  _result = _execute.execute(b"NInTwoTypeVariables", 0, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  _result = None
  return _result


@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('n_ints_in')
def n_ints_in(a: Annotated[List[Any], _atypes.Int32], name=None):
  r"""TODO: add doc.

  Args:
    a: A list of at least 2 `Tensor` objects with type `int32`.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "NIntsIn", name, a)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_n_ints_in(
          (a, name,), None)
      if _result is not NotImplemented:
        return _result
      return n_ints_in_eager_fallback(
          a, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            n_ints_in, (), dict(a=a, name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_n_ints_in(
        (a, name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  if not isinstance(a, (list, tuple)):
    raise TypeError(
        "Expected list for 'a' argument to "
        "'n_ints_in' Op, not %r." % a)
  _attr_N = len(a)
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "NIntsIn", a=a, name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          n_ints_in, (), dict(a=a, name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  return _op
NIntsIn = tf_export("raw_ops.NIntsIn")(_ops.to_raw_op(n_ints_in))
_dispatcher_for_n_ints_in = n_ints_in._tf_type_based_dispatcher.Dispatch


def n_ints_in_eager_fallback(a: Annotated[List[Any], _atypes.Int32], name, ctx):
  if not isinstance(a, (list, tuple)):
    raise TypeError(
        "Expected list for 'a' argument to "
        "'n_ints_in' Op, not %r." % a)
  _attr_N = len(a)
  a = _ops.convert_n_to_tensor(a, _dtypes.int32)
  _inputs_flat = list(a)
  _attrs = ("N", _attr_N)
  _result = _execute.execute(b"NIntsIn", 0, inputs=_inputs_flat, attrs=_attrs,
                             ctx=ctx, name=name)
  _result = None
  return _result


@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('n_ints_out')
def n_ints_out(N: int, name=None):
  r"""TODO: add doc.

  Args:
    N: An `int` that is `>= 2`.
    name: A name for the operation (optional).

  Returns:
    A list of `N` `Tensor` objects with type `int32`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "NIntsOut", name, "N", N)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_n_ints_out(
          (N, name,), None)
      if _result is not NotImplemented:
        return _result
      return n_ints_out_eager_fallback(
          N=N, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            n_ints_out, (), dict(N=N, name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_n_ints_out(
        (N, name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  N = _execute.make_int(N, "N")
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "NIntsOut", N=N, name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          n_ints_out, (), dict(N=N, name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("N", _op._get_attr_int("N"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "NIntsOut", _inputs_flat, _attrs, _result)
  return _result

NIntsOut = tf_export("raw_ops.NIntsOut")(_ops.to_raw_op(n_ints_out))
_dispatcher_for_n_ints_out = n_ints_out._tf_type_based_dispatcher.Dispatch


def n_ints_out_eager_fallback(N: int, name, ctx):
  N = _execute.make_int(N, "N")
  _inputs_flat = []
  _attrs = ("N", N)
  _result = _execute.execute(b"NIntsOut", N, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "NIntsOut", _inputs_flat, _attrs, _result)
  return _result


@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('n_ints_out_default')
def n_ints_out_default(N:int=3, name=None):
  r"""TODO: add doc.

  Args:
    N: An optional `int` that is `>= 2`. Defaults to `3`.
    name: A name for the operation (optional).

  Returns:
    A list of `N` `Tensor` objects with type `int32`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "NIntsOutDefault", name, "N", N)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_n_ints_out_default(
          (N, name,), None)
      if _result is not NotImplemented:
        return _result
      return n_ints_out_default_eager_fallback(
          N=N, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            n_ints_out_default, (), dict(N=N, name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_n_ints_out_default(
        (N, name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  if N is None:
    N = 3
  N = _execute.make_int(N, "N")
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "NIntsOutDefault", N=N, name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          n_ints_out_default, (), dict(N=N, name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("N", _op._get_attr_int("N"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "NIntsOutDefault", _inputs_flat, _attrs, _result)
  return _result

NIntsOutDefault = tf_export("raw_ops.NIntsOutDefault")(_ops.to_raw_op(n_ints_out_default))
_dispatcher_for_n_ints_out_default = n_ints_out_default._tf_type_based_dispatcher.Dispatch


def n_ints_out_default_eager_fallback(N: int, name, ctx):
  if N is None:
    N = 3
  N = _execute.make_int(N, "N")
  _inputs_flat = []
  _attrs = ("N", N)
  _result = _execute.execute(b"NIntsOutDefault", N, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "NIntsOutDefault", _inputs_flat, _attrs, _result)
  return _result


TV_NPolymorphicIn_T = TypeVar("TV_NPolymorphicIn_T", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float16, _atypes.Float32, _atypes.Float64, _atypes.Float8e4m3fn, _atypes.Float8e5m2, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int4, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.Resource, _atypes.String, _atypes.UInt16, _atypes.UInt32, _atypes.UInt4, _atypes.UInt64, _atypes.UInt8, _atypes.Variant)

@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('n_polymorphic_in')
def n_polymorphic_in(a: Annotated[List[Any], TV_NPolymorphicIn_T], name=None):
  r"""TODO: add doc.

  Args:
    a: A list of at least 2 `Tensor` objects with the same type.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "NPolymorphicIn", name, a)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_n_polymorphic_in(
          (a, name,), None)
      if _result is not NotImplemented:
        return _result
      return n_polymorphic_in_eager_fallback(
          a, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            n_polymorphic_in, (), dict(a=a, name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_n_polymorphic_in(
        (a, name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  if not isinstance(a, (list, tuple)):
    raise TypeError(
        "Expected list for 'a' argument to "
        "'n_polymorphic_in' Op, not %r." % a)
  _attr_N = len(a)
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "NPolymorphicIn", a=a, name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          n_polymorphic_in, (), dict(a=a, name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  return _op
NPolymorphicIn = tf_export("raw_ops.NPolymorphicIn")(_ops.to_raw_op(n_polymorphic_in))
_dispatcher_for_n_polymorphic_in = n_polymorphic_in._tf_type_based_dispatcher.Dispatch


def n_polymorphic_in_eager_fallback(a: Annotated[List[Any], TV_NPolymorphicIn_T], name, ctx):
  if not isinstance(a, (list, tuple)):
    raise TypeError(
        "Expected list for 'a' argument to "
        "'n_polymorphic_in' Op, not %r." % a)
  _attr_N = len(a)
  _attr_T, a = _execute.args_to_matching_eager(list(a), ctx, [])
  _inputs_flat = list(a)
  _attrs = ("T", _attr_T, "N", _attr_N)
  _result = _execute.execute(b"NPolymorphicIn", 0, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  _result = None
  return _result


TV_NPolymorphicOut_T = TypeVar("TV_NPolymorphicOut_T", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float16, _atypes.Float32, _atypes.Float64, _atypes.Float8e4m3fn, _atypes.Float8e5m2, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int4, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.Resource, _atypes.String, _atypes.UInt16, _atypes.UInt32, _atypes.UInt4, _atypes.UInt64, _atypes.UInt8, _atypes.Variant)

@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('n_polymorphic_out')
def n_polymorphic_out(T: TV_NPolymorphicOut_T, N: int, name=None):
  r"""TODO: add doc.

  Args:
    T: A `tf.DType`.
    N: An `int` that is `>= 2`.
    name: A name for the operation (optional).

  Returns:
    A list of `N` `Tensor` objects with type `T`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "NPolymorphicOut", name, "T", T, "N", N)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_n_polymorphic_out(
          (T, N, name,), None)
      if _result is not NotImplemented:
        return _result
      return n_polymorphic_out_eager_fallback(
          T=T, N=N, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            n_polymorphic_out, (), dict(T=T, N=N, name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_n_polymorphic_out(
        (T, N, name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  T = _execute.make_type(T, "T")
  N = _execute.make_int(N, "N")
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "NPolymorphicOut", T=T, N=N, name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          n_polymorphic_out, (), dict(T=T, N=N, name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"), "N", _op._get_attr_int("N"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "NPolymorphicOut", _inputs_flat, _attrs, _result)
  return _result

NPolymorphicOut = tf_export("raw_ops.NPolymorphicOut")(_ops.to_raw_op(n_polymorphic_out))
_dispatcher_for_n_polymorphic_out = n_polymorphic_out._tf_type_based_dispatcher.Dispatch


def n_polymorphic_out_eager_fallback(T: TV_NPolymorphicOut_T, N: int, name, ctx):
  T = _execute.make_type(T, "T")
  N = _execute.make_int(N, "N")
  _inputs_flat = []
  _attrs = ("T", T, "N", N)
  _result = _execute.execute(b"NPolymorphicOut", N, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "NPolymorphicOut", _inputs_flat, _attrs, _result)
  return _result


TV_NPolymorphicOutDefault_T = TypeVar("TV_NPolymorphicOutDefault_T", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float16, _atypes.Float32, _atypes.Float64, _atypes.Float8e4m3fn, _atypes.Float8e5m2, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int4, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.Resource, _atypes.String, _atypes.UInt16, _atypes.UInt32, _atypes.UInt4, _atypes.UInt64, _atypes.UInt8, _atypes.Variant)

@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('n_polymorphic_out_default')
def n_polymorphic_out_default(T:TV_NPolymorphicOutDefault_T=_dtypes.bool, N:int=2, name=None):
  r"""TODO: add doc.

  Args:
    T: An optional `tf.DType`. Defaults to `tf.bool`.
    N: An optional `int` that is `>= 2`. Defaults to `2`.
    name: A name for the operation (optional).

  Returns:
    A list of `N` `Tensor` objects with type `T`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "NPolymorphicOutDefault", name, "T", T, "N", N)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_n_polymorphic_out_default(
          (T, N, name,), None)
      if _result is not NotImplemented:
        return _result
      return n_polymorphic_out_default_eager_fallback(
          T=T, N=N, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            n_polymorphic_out_default, (), dict(T=T, N=N, name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_n_polymorphic_out_default(
        (T, N, name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  if T is None:
    T = _dtypes.bool
  T = _execute.make_type(T, "T")
  if N is None:
    N = 2
  N = _execute.make_int(N, "N")
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "NPolymorphicOutDefault", T=T, N=N, name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          n_polymorphic_out_default, (), dict(T=T, N=N, name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"), "N", _op._get_attr_int("N"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "NPolymorphicOutDefault", _inputs_flat, _attrs, _result)
  return _result

NPolymorphicOutDefault = tf_export("raw_ops.NPolymorphicOutDefault")(_ops.to_raw_op(n_polymorphic_out_default))
_dispatcher_for_n_polymorphic_out_default = n_polymorphic_out_default._tf_type_based_dispatcher.Dispatch


def n_polymorphic_out_default_eager_fallback(T: TV_NPolymorphicOutDefault_T, N: int, name, ctx):
  if T is None:
    T = _dtypes.bool
  T = _execute.make_type(T, "T")
  if N is None:
    N = 2
  N = _execute.make_int(N, "N")
  _inputs_flat = []
  _attrs = ("T", T, "N", N)
  _result = _execute.execute(b"NPolymorphicOutDefault", N,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "NPolymorphicOutDefault", _inputs_flat, _attrs, _result)
  return _result


TV_NPolymorphicRestrictIn_T = TypeVar("TV_NPolymorphicRestrictIn_T", _atypes.Bool, _atypes.String)

@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('n_polymorphic_restrict_in')
def n_polymorphic_restrict_in(a: Annotated[List[Any], TV_NPolymorphicRestrictIn_T], name=None):
  r"""TODO: add doc.

  Args:
    a: A list of at least 2 `Tensor` objects with the same type in: `string`, `bool`.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "NPolymorphicRestrictIn", name, a)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_n_polymorphic_restrict_in(
          (a, name,), None)
      if _result is not NotImplemented:
        return _result
      return n_polymorphic_restrict_in_eager_fallback(
          a, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            n_polymorphic_restrict_in, (), dict(a=a, name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_n_polymorphic_restrict_in(
        (a, name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  if not isinstance(a, (list, tuple)):
    raise TypeError(
        "Expected list for 'a' argument to "
        "'n_polymorphic_restrict_in' Op, not %r." % a)
  _attr_N = len(a)
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "NPolymorphicRestrictIn", a=a, name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          n_polymorphic_restrict_in, (), dict(a=a, name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  return _op
NPolymorphicRestrictIn = tf_export("raw_ops.NPolymorphicRestrictIn")(_ops.to_raw_op(n_polymorphic_restrict_in))
_dispatcher_for_n_polymorphic_restrict_in = n_polymorphic_restrict_in._tf_type_based_dispatcher.Dispatch


def n_polymorphic_restrict_in_eager_fallback(a: Annotated[List[Any], TV_NPolymorphicRestrictIn_T], name, ctx):
  if not isinstance(a, (list, tuple)):
    raise TypeError(
        "Expected list for 'a' argument to "
        "'n_polymorphic_restrict_in' Op, not %r." % a)
  _attr_N = len(a)
  _attr_T, a = _execute.args_to_matching_eager(list(a), ctx, [_dtypes.string, _dtypes.bool, ])
  _inputs_flat = list(a)
  _attrs = ("T", _attr_T, "N", _attr_N)
  _result = _execute.execute(b"NPolymorphicRestrictIn", 0,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  _result = None
  return _result


TV_NPolymorphicRestrictOut_T = TypeVar("TV_NPolymorphicRestrictOut_T", _atypes.Bool, _atypes.String)

@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('n_polymorphic_restrict_out')
def n_polymorphic_restrict_out(T: TV_NPolymorphicRestrictOut_T, N: int, name=None):
  r"""TODO: add doc.

  Args:
    T: A `tf.DType` from: `tf.string, tf.bool`.
    N: An `int` that is `>= 2`.
    name: A name for the operation (optional).

  Returns:
    A list of `N` `Tensor` objects with type `T`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "NPolymorphicRestrictOut", name, "T", T, "N", N)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_n_polymorphic_restrict_out(
          (T, N, name,), None)
      if _result is not NotImplemented:
        return _result
      return n_polymorphic_restrict_out_eager_fallback(
          T=T, N=N, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            n_polymorphic_restrict_out, (), dict(T=T, N=N, name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_n_polymorphic_restrict_out(
        (T, N, name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  T = _execute.make_type(T, "T")
  N = _execute.make_int(N, "N")
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "NPolymorphicRestrictOut", T=T, N=N, name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          n_polymorphic_restrict_out, (), dict(T=T, N=N, name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"), "N", _op._get_attr_int("N"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "NPolymorphicRestrictOut", _inputs_flat, _attrs, _result)
  return _result

NPolymorphicRestrictOut = tf_export("raw_ops.NPolymorphicRestrictOut")(_ops.to_raw_op(n_polymorphic_restrict_out))
_dispatcher_for_n_polymorphic_restrict_out = n_polymorphic_restrict_out._tf_type_based_dispatcher.Dispatch


def n_polymorphic_restrict_out_eager_fallback(T: TV_NPolymorphicRestrictOut_T, N: int, name, ctx):
  T = _execute.make_type(T, "T")
  N = _execute.make_int(N, "N")
  _inputs_flat = []
  _attrs = ("T", T, "N", N)
  _result = _execute.execute(b"NPolymorphicRestrictOut", N,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "NPolymorphicRestrictOut", _inputs_flat, _attrs, _result)
  return _result

_Namespace_TestStringOutputOutput = collections.namedtuple(
    "Namespace_TestStringOutput",
    ["output1", "output2"])


@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('namespace_test_string_output')
def namespace_test_string_output(input: Annotated[Any, _atypes.Float32], name=None):
  r"""TODO: add doc.

  Args:
    input: A `Tensor` of type `float32`.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (output1, output2).

    output1: A `Tensor` of type `float32`.
    output2: A `Tensor` of type `string`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "Namespace>TestStringOutput", name, input)
      _result = _Namespace_TestStringOutputOutput._make(_result)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_namespace_test_string_output(
          (input, name,), None)
      if _result is not NotImplemented:
        return _result
      return namespace_test_string_output_eager_fallback(
          input, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            namespace_test_string_output, (), dict(input=input, name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_namespace_test_string_output(
        (input, name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "Namespace>TestStringOutput", input=input, name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          namespace_test_string_output, (), dict(input=input, name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ()
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "Namespace>TestStringOutput", _inputs_flat, _attrs, _result)
  _result = _Namespace_TestStringOutputOutput._make(_result)
  return _result

Namespace_TestStringOutput = tf_export("raw_ops.Namespace_TestStringOutput")(_ops.to_raw_op(namespace_test_string_output))
_dispatcher_for_namespace_test_string_output = namespace_test_string_output._tf_type_based_dispatcher.Dispatch


def namespace_test_string_output_eager_fallback(input: Annotated[Any, _atypes.Float32], name, ctx):
  input = _ops.convert_to_tensor(input, _dtypes.float32)
  _inputs_flat = [input]
  _attrs = None
  _result = _execute.execute(b"Namespace>TestStringOutput", 2,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "Namespace>TestStringOutput", _inputs_flat, _attrs, _result)
  _result = _Namespace_TestStringOutputOutput._make(_result)
  return _result


@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('none')
def none(name=None):
  r"""TODO: add doc.

  Args:
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "None", name)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_none(
          (name,), None)
      if _result is not NotImplemented:
        return _result
      return none_eager_fallback(
          name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            none, (), dict(name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_none(
        (name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "None", name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          none, (), dict(name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  return _op
None_ = tf_export("raw_ops.None_")(_ops.to_raw_op(none))
_dispatcher_for_none = none._tf_type_based_dispatcher.Dispatch


def none_eager_fallback(name, ctx):
  _inputs_flat = []
  _attrs = None
  _result = _execute.execute(b"None", 0, inputs=_inputs_flat, attrs=_attrs,
                             ctx=ctx, name=name)
  _result = None
  return _result


@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('old')
def old(name=None):
  r"""TODO: add doc.

  Args:
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "Old", name)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_old(
          (name,), None)
      if _result is not NotImplemented:
        return _result
      return old_eager_fallback(
          name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            old, (), dict(name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_old(
        (name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "Old", name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          old, (), dict(name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  return _op
Old = tf_export("raw_ops.Old")(_ops.to_raw_op(old))
_dispatcher_for_old = old._tf_type_based_dispatcher.Dispatch


def old_eager_fallback(name, ctx):
  _inputs_flat = []
  _attrs = None
  _result = _execute.execute(b"Old", 0, inputs=_inputs_flat, attrs=_attrs,
                             ctx=ctx, name=name)
  _result = None
  return _result


@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('op_with_default_attr')
def op_with_default_attr(default_float:float=123, name=None) -> Annotated[Any, _atypes.Int32]:
  r"""TODO: add doc.

  Args:
    default_float: An optional `float`. Defaults to `123`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `int32`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "OpWithDefaultAttr", name, "default_float", default_float)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_op_with_default_attr(
          (default_float, name,), None)
      if _result is not NotImplemented:
        return _result
      return op_with_default_attr_eager_fallback(
          default_float=default_float, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            op_with_default_attr, (), dict(default_float=default_float,
                                           name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_op_with_default_attr(
        (default_float, name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  if default_float is None:
    default_float = 123
  default_float = _execute.make_float(default_float, "default_float")
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "OpWithDefaultAttr", default_float=default_float, name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          op_with_default_attr, (), dict(default_float=default_float,
                                         name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("default_float", _op.get_attr("default_float"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "OpWithDefaultAttr", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

OpWithDefaultAttr = tf_export("raw_ops.OpWithDefaultAttr")(_ops.to_raw_op(op_with_default_attr))
_dispatcher_for_op_with_default_attr = op_with_default_attr._tf_type_based_dispatcher.Dispatch


def op_with_default_attr_eager_fallback(default_float: float, name, ctx) -> Annotated[Any, _atypes.Int32]:
  if default_float is None:
    default_float = 123
  default_float = _execute.make_float(default_float, "default_float")
  _inputs_flat = []
  _attrs = ("default_float", default_float)
  _result = _execute.execute(b"OpWithDefaultAttr", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "OpWithDefaultAttr", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('op_with_future_default_attr')
def op_with_future_default_attr(name=None):
  r"""TODO: add doc.

  Args:
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "OpWithFutureDefaultAttr", name)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_op_with_future_default_attr(
          (name,), None)
      if _result is not NotImplemented:
        return _result
      return op_with_future_default_attr_eager_fallback(
          name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            op_with_future_default_attr, (), dict(name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_op_with_future_default_attr(
        (name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "OpWithFutureDefaultAttr", name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          op_with_future_default_attr, (), dict(name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  return _op
OpWithFutureDefaultAttr = tf_export("raw_ops.OpWithFutureDefaultAttr")(_ops.to_raw_op(op_with_future_default_attr))
_dispatcher_for_op_with_future_default_attr = op_with_future_default_attr._tf_type_based_dispatcher.Dispatch


def op_with_future_default_attr_eager_fallback(name, ctx):
  _inputs_flat = []
  _attrs = None
  _result = _execute.execute(b"OpWithFutureDefaultAttr", 0,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  _result = None
  return _result


TV_OutT_T = TypeVar("TV_OutT_T", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float16, _atypes.Float32, _atypes.Float64, _atypes.Float8e4m3fn, _atypes.Float8e5m2, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int4, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.Resource, _atypes.String, _atypes.UInt16, _atypes.UInt32, _atypes.UInt4, _atypes.UInt64, _atypes.UInt8, _atypes.Variant)

@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('out_t')
def out_t(T: TV_OutT_T, name=None) -> Annotated[Any, TV_OutT_T]:
  r"""TODO: add doc.

  Args:
    T: A `tf.DType`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `T`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "OutT", name, "T", T)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_out_t(
          (T, name,), None)
      if _result is not NotImplemented:
        return _result
      return out_t_eager_fallback(
          T=T, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            out_t, (), dict(T=T, name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_out_t(
        (T, name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  T = _execute.make_type(T, "T")
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "OutT", T=T, name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          out_t, (), dict(T=T, name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "OutT", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

OutT = tf_export("raw_ops.OutT")(_ops.to_raw_op(out_t))
_dispatcher_for_out_t = out_t._tf_type_based_dispatcher.Dispatch


def out_t_eager_fallback(T: TV_OutT_T, name, ctx) -> Annotated[Any, TV_OutT_T]:
  T = _execute.make_type(T, "T")
  _inputs_flat = []
  _attrs = ("T", T)
  _result = _execute.execute(b"OutT", 1, inputs=_inputs_flat, attrs=_attrs,
                             ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "OutT", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('out_type_list')
def out_type_list(T, name=None):
  r"""TODO: add doc.

  Args:
    T: A list of `tf.DTypes`.
    name: A name for the operation (optional).

  Returns:
    A list of `Tensor` objects of type `T`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "OutTypeList", name, "T", T)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_out_type_list(
          (T, name,), None)
      if _result is not NotImplemented:
        return _result
      return out_type_list_eager_fallback(
          T=T, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            out_type_list, (), dict(T=T, name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_out_type_list(
        (T, name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  if not isinstance(T, (list, tuple)):
    raise TypeError(
        "Expected list for 'T' argument to "
        "'out_type_list' Op, not %r." % T)
  T = [_execute.make_type(_t, "T") for _t in T]
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "OutTypeList", T=T, name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          out_type_list, (), dict(T=T, name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op.get_attr("T"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "OutTypeList", _inputs_flat, _attrs, _result)
  return _result

OutTypeList = tf_export("raw_ops.OutTypeList")(_ops.to_raw_op(out_type_list))
_dispatcher_for_out_type_list = out_type_list._tf_type_based_dispatcher.Dispatch


def out_type_list_eager_fallback(T, name, ctx):
  if not isinstance(T, (list, tuple)):
    raise TypeError(
        "Expected list for 'T' argument to "
        "'out_type_list' Op, not %r." % T)
  T = [_execute.make_type(_t, "T") for _t in T]
  _inputs_flat = []
  _attrs = ("T", T)
  _result = _execute.execute(b"OutTypeList", len(T), inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "OutTypeList", _inputs_flat, _attrs, _result)
  return _result


@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('out_type_list_restrict')
def out_type_list_restrict(t, name=None):
  r"""TODO: add doc.

  Args:
    t: A list of `tf.DTypes` from: `tf.string, tf.bool` that has length `>= 1`.
    name: A name for the operation (optional).

  Returns:
    A list of `Tensor` objects of type `t`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "OutTypeListRestrict", name, "t", t)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_out_type_list_restrict(
          (t, name,), None)
      if _result is not NotImplemented:
        return _result
      return out_type_list_restrict_eager_fallback(
          t=t, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            out_type_list_restrict, (), dict(t=t, name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_out_type_list_restrict(
        (t, name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  if not isinstance(t, (list, tuple)):
    raise TypeError(
        "Expected list for 't' argument to "
        "'out_type_list_restrict' Op, not %r." % t)
  t = [_execute.make_type(_t, "t") for _t in t]
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "OutTypeListRestrict", t=t, name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          out_type_list_restrict, (), dict(t=t, name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("t", _op.get_attr("t"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "OutTypeListRestrict", _inputs_flat, _attrs, _result)
  return _result

OutTypeListRestrict = tf_export("raw_ops.OutTypeListRestrict")(_ops.to_raw_op(out_type_list_restrict))
_dispatcher_for_out_type_list_restrict = out_type_list_restrict._tf_type_based_dispatcher.Dispatch


def out_type_list_restrict_eager_fallback(t, name, ctx):
  if not isinstance(t, (list, tuple)):
    raise TypeError(
        "Expected list for 't' argument to "
        "'out_type_list_restrict' Op, not %r." % t)
  t = [_execute.make_type(_t, "t") for _t in t]
  _inputs_flat = []
  _attrs = ("t", t)
  _result = _execute.execute(b"OutTypeListRestrict", len(t),
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "OutTypeListRestrict", _inputs_flat, _attrs, _result)
  return _result


TV_Polymorphic_T = TypeVar("TV_Polymorphic_T", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float16, _atypes.Float32, _atypes.Float64, _atypes.Float8e4m3fn, _atypes.Float8e5m2, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int4, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.Resource, _atypes.String, _atypes.UInt16, _atypes.UInt32, _atypes.UInt4, _atypes.UInt64, _atypes.UInt8, _atypes.Variant)

@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('polymorphic')
def polymorphic(a: Annotated[Any, TV_Polymorphic_T], name=None) -> Annotated[Any, TV_Polymorphic_T]:
  r"""TODO: add doc.

  Args:
    a: A `Tensor`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `a`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "Polymorphic", name, a)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_polymorphic(
          (a, name,), None)
      if _result is not NotImplemented:
        return _result
      return polymorphic_eager_fallback(
          a, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            polymorphic, (), dict(a=a, name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_polymorphic(
        (a, name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "Polymorphic", a=a, name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          polymorphic, (), dict(a=a, name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "Polymorphic", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

Polymorphic = tf_export("raw_ops.Polymorphic")(_ops.to_raw_op(polymorphic))
_dispatcher_for_polymorphic = polymorphic._tf_type_based_dispatcher.Dispatch


def polymorphic_eager_fallback(a: Annotated[Any, TV_Polymorphic_T], name, ctx) -> Annotated[Any, TV_Polymorphic_T]:
  _attr_T, (a,) = _execute.args_to_matching_eager([a], ctx, [])
  _inputs_flat = [a]
  _attrs = ("T", _attr_T)
  _result = _execute.execute(b"Polymorphic", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "Polymorphic", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_PolymorphicDefaultOut_T = TypeVar("TV_PolymorphicDefaultOut_T", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float16, _atypes.Float32, _atypes.Float64, _atypes.Float8e4m3fn, _atypes.Float8e5m2, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int4, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.Resource, _atypes.String, _atypes.UInt16, _atypes.UInt32, _atypes.UInt4, _atypes.UInt64, _atypes.UInt8, _atypes.Variant)

@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('polymorphic_default_out')
def polymorphic_default_out(T:TV_PolymorphicDefaultOut_T=_dtypes.string, name=None) -> Annotated[Any, TV_PolymorphicDefaultOut_T]:
  r"""TODO: add doc.

  Args:
    T: An optional `tf.DType`. Defaults to `tf.string`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `T`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "PolymorphicDefaultOut", name, "T", T)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_polymorphic_default_out(
          (T, name,), None)
      if _result is not NotImplemented:
        return _result
      return polymorphic_default_out_eager_fallback(
          T=T, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            polymorphic_default_out, (), dict(T=T, name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_polymorphic_default_out(
        (T, name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  if T is None:
    T = _dtypes.string
  T = _execute.make_type(T, "T")
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "PolymorphicDefaultOut", T=T, name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          polymorphic_default_out, (), dict(T=T, name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "PolymorphicDefaultOut", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

PolymorphicDefaultOut = tf_export("raw_ops.PolymorphicDefaultOut")(_ops.to_raw_op(polymorphic_default_out))
_dispatcher_for_polymorphic_default_out = polymorphic_default_out._tf_type_based_dispatcher.Dispatch


def polymorphic_default_out_eager_fallback(T: TV_PolymorphicDefaultOut_T, name, ctx) -> Annotated[Any, TV_PolymorphicDefaultOut_T]:
  if T is None:
    T = _dtypes.string
  T = _execute.make_type(T, "T")
  _inputs_flat = []
  _attrs = ("T", T)
  _result = _execute.execute(b"PolymorphicDefaultOut", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "PolymorphicDefaultOut", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_PolymorphicOut_T = TypeVar("TV_PolymorphicOut_T", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float16, _atypes.Float32, _atypes.Float64, _atypes.Float8e4m3fn, _atypes.Float8e5m2, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int4, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.Resource, _atypes.String, _atypes.UInt16, _atypes.UInt32, _atypes.UInt4, _atypes.UInt64, _atypes.UInt8, _atypes.Variant)

@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('polymorphic_out')
def polymorphic_out(T: TV_PolymorphicOut_T, name=None) -> Annotated[Any, TV_PolymorphicOut_T]:
  r"""TODO: add doc.

  Args:
    T: A `tf.DType`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `T`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "PolymorphicOut", name, "T", T)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_polymorphic_out(
          (T, name,), None)
      if _result is not NotImplemented:
        return _result
      return polymorphic_out_eager_fallback(
          T=T, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            polymorphic_out, (), dict(T=T, name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_polymorphic_out(
        (T, name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  T = _execute.make_type(T, "T")
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "PolymorphicOut", T=T, name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          polymorphic_out, (), dict(T=T, name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "PolymorphicOut", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

PolymorphicOut = tf_export("raw_ops.PolymorphicOut")(_ops.to_raw_op(polymorphic_out))
_dispatcher_for_polymorphic_out = polymorphic_out._tf_type_based_dispatcher.Dispatch


def polymorphic_out_eager_fallback(T: TV_PolymorphicOut_T, name, ctx) -> Annotated[Any, TV_PolymorphicOut_T]:
  T = _execute.make_type(T, "T")
  _inputs_flat = []
  _attrs = ("T", T)
  _result = _execute.execute(b"PolymorphicOut", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "PolymorphicOut", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_RefIn_T = TypeVar("TV_RefIn_T", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float16, _atypes.Float32, _atypes.Float64, _atypes.Float8e4m3fn, _atypes.Float8e5m2, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int4, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.Resource, _atypes.String, _atypes.UInt16, _atypes.UInt32, _atypes.UInt4, _atypes.UInt64, _atypes.UInt8, _atypes.Variant)

@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('ref_in')
def ref_in(a: Annotated[Any, TV_RefIn_T], name=None):
  r"""TODO: add doc.

  Args:
    a: A mutable `Tensor`.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    raise RuntimeError("ref_in op does not support eager execution. Arg 'a' is a ref.")
  else:
    _result = _dispatcher_for_ref_in(
        (a, name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "RefIn", a=a, name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          ref_in, (), dict(a=a, name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  return _op
RefIn = tf_export("raw_ops.RefIn")(_ops.to_raw_op(ref_in))
_dispatcher_for_ref_in = ref_in._tf_type_based_dispatcher.Dispatch


def ref_in_eager_fallback(a: Annotated[Any, TV_RefIn_T], name, ctx):
  raise RuntimeError("ref_in op does not support eager execution. Arg 'a' is a ref.")

@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('ref_input_float_input')
def ref_input_float_input(a: Annotated[Any, _atypes.Float32], b: Annotated[Any, _atypes.Float32], name=None):
  r"""TODO: add doc.

  Args:
    a: A `Tensor` of type mutable `float32`.
    b: A `Tensor` of type `float32`.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    raise RuntimeError("ref_input_float_input op does not support eager execution. Arg 'a' is a ref.")
  else:
    _result = _dispatcher_for_ref_input_float_input(
        (a, b, name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "RefInputFloatInput", a=a, b=b, name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          ref_input_float_input, (), dict(a=a, b=b, name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  return _op
RefInputFloatInput = tf_export("raw_ops.RefInputFloatInput")(_ops.to_raw_op(ref_input_float_input))
_dispatcher_for_ref_input_float_input = ref_input_float_input._tf_type_based_dispatcher.Dispatch


def ref_input_float_input_eager_fallback(a: Annotated[Any, _atypes.Float32], b: Annotated[Any, _atypes.Float32], name, ctx):
  raise RuntimeError("ref_input_float_input op does not support eager execution. Arg 'a' is a ref.")

@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('ref_input_float_input_int_output')
def ref_input_float_input_int_output(a: Annotated[Any, _atypes.Float32], b: Annotated[Any, _atypes.Float32], name=None) -> Annotated[Any, _atypes.Int32]:
  r"""TODO: add doc.

  Args:
    a: A `Tensor` of type mutable `float32`.
    b: A `Tensor` of type `float32`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `int32`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    raise RuntimeError("ref_input_float_input_int_output op does not support eager execution. Arg 'a' is a ref.")
  else:
    _result = _dispatcher_for_ref_input_float_input_int_output(
        (a, b, name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "RefInputFloatInputIntOutput", a=a, b=b, name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          ref_input_float_input_int_output, (), dict(a=a, b=b, name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ()
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "RefInputFloatInputIntOutput", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

RefInputFloatInputIntOutput = tf_export("raw_ops.RefInputFloatInputIntOutput")(_ops.to_raw_op(ref_input_float_input_int_output))
_dispatcher_for_ref_input_float_input_int_output = ref_input_float_input_int_output._tf_type_based_dispatcher.Dispatch


def ref_input_float_input_int_output_eager_fallback(a: Annotated[Any, _atypes.Float32], b: Annotated[Any, _atypes.Float32], name, ctx) -> Annotated[Any, _atypes.Int32]:
  raise RuntimeError("ref_input_float_input_int_output op does not support eager execution. Arg 'a' is a ref.")

@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('ref_input_int_input')
def ref_input_int_input(a: Annotated[Any, _atypes.Int32], b: Annotated[Any, _atypes.Int32], name=None):
  r"""TODO: add doc.

  Args:
    a: A `Tensor` of type mutable `int32`.
    b: A `Tensor` of type `int32`.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    raise RuntimeError("ref_input_int_input op does not support eager execution. Arg 'a' is a ref.")
  else:
    _result = _dispatcher_for_ref_input_int_input(
        (a, b, name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "RefInputIntInput", a=a, b=b, name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          ref_input_int_input, (), dict(a=a, b=b, name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  return _op
RefInputIntInput = tf_export("raw_ops.RefInputIntInput")(_ops.to_raw_op(ref_input_int_input))
_dispatcher_for_ref_input_int_input = ref_input_int_input._tf_type_based_dispatcher.Dispatch


def ref_input_int_input_eager_fallback(a: Annotated[Any, _atypes.Int32], b: Annotated[Any, _atypes.Int32], name, ctx):
  raise RuntimeError("ref_input_int_input op does not support eager execution. Arg 'a' is a ref.")

TV_RefOut_T = TypeVar("TV_RefOut_T", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float16, _atypes.Float32, _atypes.Float64, _atypes.Float8e4m3fn, _atypes.Float8e5m2, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int4, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.Resource, _atypes.String, _atypes.UInt16, _atypes.UInt32, _atypes.UInt4, _atypes.UInt64, _atypes.UInt8, _atypes.Variant)

@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('ref_out')
def ref_out(T: TV_RefOut_T, name=None) -> Annotated[Any, TV_RefOut_T]:
  r"""TODO: add doc.

  Args:
    T: A `tf.DType`.
    name: A name for the operation (optional).

  Returns:
    A mutable `Tensor` of type `T`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    raise RuntimeError("ref_out op does not support eager execution. Arg 'a' is a ref.")
  else:
    _result = _dispatcher_for_ref_out(
        (T, name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  T = _execute.make_type(T, "T")
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "RefOut", T=T, name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          ref_out, (), dict(T=T, name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "RefOut", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

RefOut = tf_export("raw_ops.RefOut")(_ops.to_raw_op(ref_out))
_dispatcher_for_ref_out = ref_out._tf_type_based_dispatcher.Dispatch


def ref_out_eager_fallback(T: TV_RefOut_T, name, ctx) -> Annotated[Any, TV_RefOut_T]:
  raise RuntimeError("ref_out op does not support eager execution. Arg 'a' is a ref.")

@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('ref_output')
def ref_output(name=None) -> Annotated[Any, _atypes.Int32]:
  r"""TODO: add doc.

  Args:
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type mutable `int32`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    raise RuntimeError("ref_output op does not support eager execution. Arg 'a' is a ref.")
  else:
    _result = _dispatcher_for_ref_output(
        (name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "RefOutput", name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          ref_output, (), dict(name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ()
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "RefOutput", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

RefOutput = tf_export("raw_ops.RefOutput")(_ops.to_raw_op(ref_output))
_dispatcher_for_ref_output = ref_output._tf_type_based_dispatcher.Dispatch


def ref_output_eager_fallback(name, ctx) -> Annotated[Any, _atypes.Int32]:
  raise RuntimeError("ref_output op does not support eager execution. Arg 'a' is a ref.")
_RefOutputFloatOutputOutput = collections.namedtuple(
    "RefOutputFloatOutput",
    ["a", "b"])


@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('ref_output_float_output')
def ref_output_float_output(name=None):
  r"""TODO: add doc.

  Args:
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (a, b).

    a: A `Tensor` of type mutable `float32`.
    b: A `Tensor` of type `float32`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    raise RuntimeError("ref_output_float_output op does not support eager execution. Arg 'a' is a ref.")
  else:
    _result = _dispatcher_for_ref_output_float_output(
        (name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "RefOutputFloatOutput", name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          ref_output_float_output, (), dict(name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ()
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "RefOutputFloatOutput", _inputs_flat, _attrs, _result)
  _result = _RefOutputFloatOutputOutput._make(_result)
  return _result

RefOutputFloatOutput = tf_export("raw_ops.RefOutputFloatOutput")(_ops.to_raw_op(ref_output_float_output))
_dispatcher_for_ref_output_float_output = ref_output_float_output._tf_type_based_dispatcher.Dispatch


def ref_output_float_output_eager_fallback(name, ctx):
  raise RuntimeError("ref_output_float_output op does not support eager execution. Arg 'a' is a ref.")

@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('requires_older_graph_version')
def requires_older_graph_version(name=None) -> Annotated[Any, _atypes.Int32]:
  r"""TODO: add doc.

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
        _ctx, "RequiresOlderGraphVersion", name)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_requires_older_graph_version(
          (name,), None)
      if _result is not NotImplemented:
        return _result
      return requires_older_graph_version_eager_fallback(
          name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            requires_older_graph_version, (), dict(name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_requires_older_graph_version(
        (name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "RequiresOlderGraphVersion", name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          requires_older_graph_version, (), dict(name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ()
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "RequiresOlderGraphVersion", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

RequiresOlderGraphVersion = tf_export("raw_ops.RequiresOlderGraphVersion")(_ops.to_raw_op(requires_older_graph_version))
_dispatcher_for_requires_older_graph_version = requires_older_graph_version._tf_type_based_dispatcher.Dispatch


def requires_older_graph_version_eager_fallback(name, ctx) -> Annotated[Any, _atypes.Int32]:
  _inputs_flat = []
  _attrs = None
  _result = _execute.execute(b"RequiresOlderGraphVersion", 1,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "RequiresOlderGraphVersion", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('reserved_attr')
def reserved_attr(range: int, name=None):
  r"""TODO: add doc.

  Args:
    range: An `int`.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "ReservedAttr", name, "range", range)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_reserved_attr(
          (range, name,), None)
      if _result is not NotImplemented:
        return _result
      return reserved_attr_eager_fallback(
          range=range, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            reserved_attr, (), dict(range=range, name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_reserved_attr(
        (range, name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  range = _execute.make_int(range, "range")
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "ReservedAttr", range=range, name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          reserved_attr, (), dict(range=range, name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  return _op
ReservedAttr = tf_export("raw_ops.ReservedAttr")(_ops.to_raw_op(reserved_attr))
_dispatcher_for_reserved_attr = reserved_attr._tf_type_based_dispatcher.Dispatch


def reserved_attr_eager_fallback(range: int, name, ctx):
  range = _execute.make_int(range, "range")
  _inputs_flat = []
  _attrs = ("range", range)
  _result = _execute.execute(b"ReservedAttr", 0, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  _result = None
  return _result


@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('reserved_input')
def reserved_input(input: Annotated[Any, _atypes.Int32], name=None):
  r"""TODO: add doc.

  Args:
    input: A `Tensor` of type `int32`.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "ReservedInput", name, input)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_reserved_input(
          (input, name,), None)
      if _result is not NotImplemented:
        return _result
      return reserved_input_eager_fallback(
          input, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            reserved_input, (), dict(input=input, name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_reserved_input(
        (input, name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "ReservedInput", input=input, name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          reserved_input, (), dict(input=input, name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  return _op
ReservedInput = tf_export("raw_ops.ReservedInput")(_ops.to_raw_op(reserved_input))
_dispatcher_for_reserved_input = reserved_input._tf_type_based_dispatcher.Dispatch


def reserved_input_eager_fallback(input: Annotated[Any, _atypes.Int32], name, ctx):
  input = _ops.convert_to_tensor(input, _dtypes.int32)
  _inputs_flat = [input]
  _attrs = None
  _result = _execute.execute(b"ReservedInput", 0, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  _result = None
  return _result


@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('resource_create_op')
def resource_create_op(resource: Annotated[Any, _atypes.Resource], name=None):
  r"""TODO: add doc.

  Args:
    resource: A `Tensor` of type `resource`.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "ResourceCreateOp", name, resource)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_resource_create_op(
          (resource, name,), None)
      if _result is not NotImplemented:
        return _result
      return resource_create_op_eager_fallback(
          resource, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            resource_create_op, (), dict(resource=resource, name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_resource_create_op(
        (resource, name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "ResourceCreateOp", resource=resource, name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          resource_create_op, (), dict(resource=resource, name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  return _op
ResourceCreateOp = tf_export("raw_ops.ResourceCreateOp")(_ops.to_raw_op(resource_create_op))
_dispatcher_for_resource_create_op = resource_create_op._tf_type_based_dispatcher.Dispatch


def resource_create_op_eager_fallback(resource: Annotated[Any, _atypes.Resource], name, ctx):
  resource = _ops.convert_to_tensor(resource, _dtypes.resource)
  _inputs_flat = [resource]
  _attrs = None
  _result = _execute.execute(b"ResourceCreateOp", 0, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  _result = None
  return _result


@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('resource_initialized_op')
def resource_initialized_op(resource: Annotated[Any, _atypes.Resource], name=None) -> Annotated[Any, _atypes.Bool]:
  r"""TODO: add doc.

  Args:
    resource: A `Tensor` of type `resource`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `bool`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "ResourceInitializedOp", name, resource)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_resource_initialized_op(
          (resource, name,), None)
      if _result is not NotImplemented:
        return _result
      return resource_initialized_op_eager_fallback(
          resource, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            resource_initialized_op, (), dict(resource=resource, name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_resource_initialized_op(
        (resource, name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "ResourceInitializedOp", resource=resource, name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          resource_initialized_op, (), dict(resource=resource, name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ()
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "ResourceInitializedOp", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

ResourceInitializedOp = tf_export("raw_ops.ResourceInitializedOp")(_ops.to_raw_op(resource_initialized_op))
_dispatcher_for_resource_initialized_op = resource_initialized_op._tf_type_based_dispatcher.Dispatch


def resource_initialized_op_eager_fallback(resource: Annotated[Any, _atypes.Resource], name, ctx) -> Annotated[Any, _atypes.Bool]:
  resource = _ops.convert_to_tensor(resource, _dtypes.resource)
  _inputs_flat = [resource]
  _attrs = None
  _result = _execute.execute(b"ResourceInitializedOp", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "ResourceInitializedOp", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('resource_using_op')
def resource_using_op(resource: Annotated[Any, _atypes.Resource], name=None):
  r"""TODO: add doc.

  Args:
    resource: A `Tensor` of type `resource`.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "ResourceUsingOp", name, resource)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_resource_using_op(
          (resource, name,), None)
      if _result is not NotImplemented:
        return _result
      return resource_using_op_eager_fallback(
          resource, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            resource_using_op, (), dict(resource=resource, name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_resource_using_op(
        (resource, name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "ResourceUsingOp", resource=resource, name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          resource_using_op, (), dict(resource=resource, name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  return _op
ResourceUsingOp = tf_export("raw_ops.ResourceUsingOp")(_ops.to_raw_op(resource_using_op))
_dispatcher_for_resource_using_op = resource_using_op._tf_type_based_dispatcher.Dispatch


def resource_using_op_eager_fallback(resource: Annotated[Any, _atypes.Resource], name, ctx):
  resource = _ops.convert_to_tensor(resource, _dtypes.resource)
  _inputs_flat = [resource]
  _attrs = None
  _result = _execute.execute(b"ResourceUsingOp", 0, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  _result = None
  return _result


TV_Restrict_T = TypeVar("TV_Restrict_T", _atypes.Bool, _atypes.String)

@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('restrict')
def restrict(a: Annotated[Any, TV_Restrict_T], name=None) -> Annotated[Any, TV_Restrict_T]:
  r"""TODO: add doc.

  Args:
    a: A `Tensor`. Must be one of the following types: `string`, `bool`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `a`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "Restrict", name, a)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_restrict(
          (a, name,), None)
      if _result is not NotImplemented:
        return _result
      return restrict_eager_fallback(
          a, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            restrict, (), dict(a=a, name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_restrict(
        (a, name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "Restrict", a=a, name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          restrict, (), dict(a=a, name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "Restrict", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

Restrict = tf_export("raw_ops.Restrict")(_ops.to_raw_op(restrict))
_dispatcher_for_restrict = restrict._tf_type_based_dispatcher.Dispatch


def restrict_eager_fallback(a: Annotated[Any, TV_Restrict_T], name, ctx) -> Annotated[Any, TV_Restrict_T]:
  _attr_T, (a,) = _execute.args_to_matching_eager([a], ctx, [_dtypes.string, _dtypes.bool, ])
  _inputs_flat = [a]
  _attrs = ("T", _attr_T)
  _result = _execute.execute(b"Restrict", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "Restrict", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('simple')
def simple(a: Annotated[Any, _atypes.Int32], name=None) -> Annotated[Any, _atypes.Float32]:
  r"""TODO: add doc.

  Args:
    a: A `Tensor` of type `int32`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `float32`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "Simple", name, a)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_simple(
          (a, name,), None)
      if _result is not NotImplemented:
        return _result
      return simple_eager_fallback(
          a, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            simple, (), dict(a=a, name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_simple(
        (a, name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "Simple", a=a, name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          simple, (), dict(a=a, name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ()
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "Simple", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

Simple = tf_export("raw_ops.Simple")(_ops.to_raw_op(simple))
_dispatcher_for_simple = simple._tf_type_based_dispatcher.Dispatch


def simple_eager_fallback(a: Annotated[Any, _atypes.Int32], name, ctx) -> Annotated[Any, _atypes.Float32]:
  a = _ops.convert_to_tensor(a, _dtypes.int32)
  _inputs_flat = [a]
  _attrs = None
  _result = _execute.execute(b"Simple", 1, inputs=_inputs_flat, attrs=_attrs,
                             ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "Simple", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('simple_struct')
def simple_struct(n_a: int, name=None):
  r"""TODO: add doc.

  Args:
    n_a: An `int` that is `>= 0`.
    name: A name for the operation (optional).

  Returns:
    A list of `n_a` `Tensor` objects with type `int32`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "SimpleStruct", name, "n_a", n_a)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_simple_struct(
          (n_a, name,), None)
      if _result is not NotImplemented:
        return _result
      return simple_struct_eager_fallback(
          n_a=n_a, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            simple_struct, (), dict(n_a=n_a, name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_simple_struct(
        (n_a, name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  n_a = _execute.make_int(n_a, "n_a")
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "SimpleStruct", n_a=n_a, name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          simple_struct, (), dict(n_a=n_a, name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("n_a", _op._get_attr_int("n_a"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "SimpleStruct", _inputs_flat, _attrs, _result)
  return _result

SimpleStruct = tf_export("raw_ops.SimpleStruct")(_ops.to_raw_op(simple_struct))
_dispatcher_for_simple_struct = simple_struct._tf_type_based_dispatcher.Dispatch


def simple_struct_eager_fallback(n_a: int, name, ctx):
  n_a = _execute.make_int(n_a, "n_a")
  _inputs_flat = []
  _attrs = ("n_a", n_a)
  _result = _execute.execute(b"SimpleStruct", n_a, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "SimpleStruct", _inputs_flat, _attrs, _result)
  return _result


TV_SleepIdentityOp_T = TypeVar("TV_SleepIdentityOp_T", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float16, _atypes.Float32, _atypes.Float64, _atypes.Float8e4m3fn, _atypes.Float8e5m2, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int4, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.Resource, _atypes.String, _atypes.UInt16, _atypes.UInt32, _atypes.UInt4, _atypes.UInt64, _atypes.UInt8, _atypes.Variant)

@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('sleep_identity_op')
def sleep_identity_op(sleep_seconds: Annotated[Any, _atypes.Int32], input: Annotated[Any, TV_SleepIdentityOp_T], name=None) -> Annotated[Any, TV_SleepIdentityOp_T]:
  r"""TODO: add doc.

  Args:
    sleep_seconds: A `Tensor` of type `int32`.
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
        _ctx, "SleepIdentityOp", name, sleep_seconds, input)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_sleep_identity_op(
          (sleep_seconds, input, name,), None)
      if _result is not NotImplemented:
        return _result
      return sleep_identity_op_eager_fallback(
          sleep_seconds, input, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            sleep_identity_op, (), dict(sleep_seconds=sleep_seconds,
                                        input=input, name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_sleep_identity_op(
        (sleep_seconds, input, name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "SleepIdentityOp", sleep_seconds=sleep_seconds, input=input,
                           name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          sleep_identity_op, (), dict(sleep_seconds=sleep_seconds,
                                      input=input, name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "SleepIdentityOp", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

SleepIdentityOp = tf_export("raw_ops.SleepIdentityOp")(_ops.to_raw_op(sleep_identity_op))
_dispatcher_for_sleep_identity_op = sleep_identity_op._tf_type_based_dispatcher.Dispatch


def sleep_identity_op_eager_fallback(sleep_seconds: Annotated[Any, _atypes.Int32], input: Annotated[Any, TV_SleepIdentityOp_T], name, ctx) -> Annotated[Any, TV_SleepIdentityOp_T]:
  _attr_T, (input,) = _execute.args_to_matching_eager([input], ctx, [])
  sleep_seconds = _ops.convert_to_tensor(sleep_seconds, _dtypes.int32)
  _inputs_flat = [sleep_seconds, input]
  _attrs = ("T", _attr_T)
  _result = _execute.execute(b"SleepIdentityOp", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "SleepIdentityOp", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('sleep_op')
def sleep_op(sleep_seconds: Annotated[Any, _atypes.Int32], name=None):
  r"""TODO: add doc.

  Args:
    sleep_seconds: A `Tensor` of type `int32`.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "SleepOp", name, sleep_seconds)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_sleep_op(
          (sleep_seconds, name,), None)
      if _result is not NotImplemented:
        return _result
      return sleep_op_eager_fallback(
          sleep_seconds, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            sleep_op, (), dict(sleep_seconds=sleep_seconds, name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_sleep_op(
        (sleep_seconds, name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "SleepOp", sleep_seconds=sleep_seconds, name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          sleep_op, (), dict(sleep_seconds=sleep_seconds, name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  return _op
SleepOp = tf_export("raw_ops.SleepOp")(_ops.to_raw_op(sleep_op))
_dispatcher_for_sleep_op = sleep_op._tf_type_based_dispatcher.Dispatch


def sleep_op_eager_fallback(sleep_seconds: Annotated[Any, _atypes.Int32], name, ctx):
  sleep_seconds = _ops.convert_to_tensor(sleep_seconds, _dtypes.int32)
  _inputs_flat = [sleep_seconds]
  _attrs = None
  _result = _execute.execute(b"SleepOp", 0, inputs=_inputs_flat, attrs=_attrs,
                             ctx=ctx, name=name)
  _result = None
  return _result


@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('string_list_attr')
def string_list_attr(a, b: str, name=None):
  r"""TODO: add doc.

  Args:
    a: A list of `strings`.
    b: A `string`.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "StringListAttr", name, "a", a, "b", b)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_string_list_attr(
          (a, b, name,), None)
      if _result is not NotImplemented:
        return _result
      return string_list_attr_eager_fallback(
          a=a, b=b, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            string_list_attr, (), dict(a=a, b=b, name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_string_list_attr(
        (a, b, name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  if not isinstance(a, (list, tuple)):
    raise TypeError(
        "Expected list for 'a' argument to "
        "'string_list_attr' Op, not %r." % a)
  a = [_execute.make_str(_s, "a") for _s in a]
  b = _execute.make_str(b, "b")
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "StringListAttr", a=a, b=b, name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          string_list_attr, (), dict(a=a, b=b, name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  return _op
StringListAttr = tf_export("raw_ops.StringListAttr")(_ops.to_raw_op(string_list_attr))
_dispatcher_for_string_list_attr = string_list_attr._tf_type_based_dispatcher.Dispatch


def string_list_attr_eager_fallback(a, b: str, name, ctx):
  if not isinstance(a, (list, tuple)):
    raise TypeError(
        "Expected list for 'a' argument to "
        "'string_list_attr' Op, not %r." % a)
  a = [_execute.make_str(_s, "a") for _s in a]
  b = _execute.make_str(b, "b")
  _inputs_flat = []
  _attrs = ("a", a, "b", b)
  _result = _execute.execute(b"StringListAttr", 0, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  _result = None
  return _result


@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('stub_resource_handle_op')
def stub_resource_handle_op(container:str="", shared_name:str="", name=None) -> Annotated[Any, _atypes.Resource]:
  r"""TODO: add doc.

  Args:
    container: An optional `string`. Defaults to `""`.
    shared_name: An optional `string`. Defaults to `""`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `resource`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "StubResourceHandleOp", name, "container", container,
        "shared_name", shared_name)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_stub_resource_handle_op(
          (container, shared_name, name,), None)
      if _result is not NotImplemented:
        return _result
      return stub_resource_handle_op_eager_fallback(
          container=container, shared_name=shared_name, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            stub_resource_handle_op, (), dict(container=container,
                                              shared_name=shared_name,
                                              name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_stub_resource_handle_op(
        (container, shared_name, name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  if container is None:
    container = ""
  container = _execute.make_str(container, "container")
  if shared_name is None:
    shared_name = ""
  shared_name = _execute.make_str(shared_name, "shared_name")
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "StubResourceHandleOp", container=container, shared_name=shared_name,
                                name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          stub_resource_handle_op, (), dict(container=container,
                                            shared_name=shared_name,
                                            name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("container", _op.get_attr("container"), "shared_name",
              _op.get_attr("shared_name"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "StubResourceHandleOp", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

StubResourceHandleOp = tf_export("raw_ops.StubResourceHandleOp")(_ops.to_raw_op(stub_resource_handle_op))
_dispatcher_for_stub_resource_handle_op = stub_resource_handle_op._tf_type_based_dispatcher.Dispatch


def stub_resource_handle_op_eager_fallback(container: str, shared_name: str, name, ctx) -> Annotated[Any, _atypes.Resource]:
  if container is None:
    container = ""
  container = _execute.make_str(container, "container")
  if shared_name is None:
    shared_name = ""
  shared_name = _execute.make_str(shared_name, "shared_name")
  _inputs_flat = []
  _attrs = ("container", container, "shared_name", shared_name)
  _result = _execute.execute(b"StubResourceHandleOp", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "StubResourceHandleOp", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_TestAttr_T = TypeVar("TV_TestAttr_T", _atypes.Float32, _atypes.Float64)

@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('test_attr')
def test_attr(T: TV_TestAttr_T, name=None) -> Annotated[Any, TV_TestAttr_T]:
  r"""TODO: add doc.

  Args:
    T: A `tf.DType` from: `tf.float32, tf.float64`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `T`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "TestAttr", name, "T", T)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_test_attr(
          (T, name,), None)
      if _result is not NotImplemented:
        return _result
      return test_attr_eager_fallback(
          T=T, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            test_attr, (), dict(T=T, name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_test_attr(
        (T, name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  T = _execute.make_type(T, "T")
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "TestAttr", T=T, name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          test_attr, (), dict(T=T, name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "TestAttr", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

TestAttr = tf_export("raw_ops.TestAttr")(_ops.to_raw_op(test_attr))
_dispatcher_for_test_attr = test_attr._tf_type_based_dispatcher.Dispatch


def test_attr_eager_fallback(T: TV_TestAttr_T, name, ctx) -> Annotated[Any, TV_TestAttr_T]:
  T = _execute.make_type(T, "T")
  _inputs_flat = []
  _attrs = ("T", T)
  _result = _execute.execute(b"TestAttr", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "TestAttr", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

_TestStringOutputOutput = collections.namedtuple(
    "TestStringOutput",
    ["output1", "output2"])


@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('test_string_output')
def test_string_output(input: Annotated[Any, _atypes.Float32], name=None):
  r"""TODO: add doc.

  Args:
    input: A `Tensor` of type `float32`.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (output1, output2).

    output1: A `Tensor` of type `float32`.
    output2: A `Tensor` of type `string`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "TestStringOutput", name, input)
      _result = _TestStringOutputOutput._make(_result)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_test_string_output(
          (input, name,), None)
      if _result is not NotImplemented:
        return _result
      return test_string_output_eager_fallback(
          input, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            test_string_output, (), dict(input=input, name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_test_string_output(
        (input, name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "TestStringOutput", input=input, name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          test_string_output, (), dict(input=input, name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ()
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "TestStringOutput", _inputs_flat, _attrs, _result)
  _result = _TestStringOutputOutput._make(_result)
  return _result

TestStringOutput = tf_export("raw_ops.TestStringOutput")(_ops.to_raw_op(test_string_output))
_dispatcher_for_test_string_output = test_string_output._tf_type_based_dispatcher.Dispatch


def test_string_output_eager_fallback(input: Annotated[Any, _atypes.Float32], name, ctx):
  input = _ops.convert_to_tensor(input, _dtypes.float32)
  _inputs_flat = [input]
  _attrs = None
  _result = _execute.execute(b"TestStringOutput", 2, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "TestStringOutput", _inputs_flat, _attrs, _result)
  _result = _TestStringOutputOutput._make(_result)
  return _result


@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('two_float_inputs')
def two_float_inputs(a: Annotated[Any, _atypes.Float32], b: Annotated[Any, _atypes.Float32], name=None):
  r"""TODO: add doc.

  Args:
    a: A `Tensor` of type `float32`.
    b: A `Tensor` of type `float32`.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "TwoFloatInputs", name, a, b)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_two_float_inputs(
          (a, b, name,), None)
      if _result is not NotImplemented:
        return _result
      return two_float_inputs_eager_fallback(
          a, b, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            two_float_inputs, (), dict(a=a, b=b, name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_two_float_inputs(
        (a, b, name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "TwoFloatInputs", a=a, b=b, name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          two_float_inputs, (), dict(a=a, b=b, name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  return _op
TwoFloatInputs = tf_export("raw_ops.TwoFloatInputs")(_ops.to_raw_op(two_float_inputs))
_dispatcher_for_two_float_inputs = two_float_inputs._tf_type_based_dispatcher.Dispatch


def two_float_inputs_eager_fallback(a: Annotated[Any, _atypes.Float32], b: Annotated[Any, _atypes.Float32], name, ctx):
  a = _ops.convert_to_tensor(a, _dtypes.float32)
  b = _ops.convert_to_tensor(b, _dtypes.float32)
  _inputs_flat = [a, b]
  _attrs = None
  _result = _execute.execute(b"TwoFloatInputs", 0, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  _result = None
  return _result


@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('two_float_inputs_float_output')
def two_float_inputs_float_output(a: Annotated[Any, _atypes.Float32], b: Annotated[Any, _atypes.Float32], name=None) -> Annotated[Any, _atypes.Float32]:
  r"""TODO: add doc.

  Args:
    a: A `Tensor` of type `float32`.
    b: A `Tensor` of type `float32`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `float32`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "TwoFloatInputsFloatOutput", name, a, b)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_two_float_inputs_float_output(
          (a, b, name,), None)
      if _result is not NotImplemented:
        return _result
      return two_float_inputs_float_output_eager_fallback(
          a, b, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            two_float_inputs_float_output, (), dict(a=a, b=b, name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_two_float_inputs_float_output(
        (a, b, name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "TwoFloatInputsFloatOutput", a=a, b=b, name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          two_float_inputs_float_output, (), dict(a=a, b=b, name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ()
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "TwoFloatInputsFloatOutput", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

TwoFloatInputsFloatOutput = tf_export("raw_ops.TwoFloatInputsFloatOutput")(_ops.to_raw_op(two_float_inputs_float_output))
_dispatcher_for_two_float_inputs_float_output = two_float_inputs_float_output._tf_type_based_dispatcher.Dispatch


def two_float_inputs_float_output_eager_fallback(a: Annotated[Any, _atypes.Float32], b: Annotated[Any, _atypes.Float32], name, ctx) -> Annotated[Any, _atypes.Float32]:
  a = _ops.convert_to_tensor(a, _dtypes.float32)
  b = _ops.convert_to_tensor(b, _dtypes.float32)
  _inputs_flat = [a, b]
  _attrs = None
  _result = _execute.execute(b"TwoFloatInputsFloatOutput", 1,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "TwoFloatInputsFloatOutput", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('two_float_inputs_int_output')
def two_float_inputs_int_output(a: Annotated[Any, _atypes.Float32], b: Annotated[Any, _atypes.Float32], name=None) -> Annotated[Any, _atypes.Int32]:
  r"""TODO: add doc.

  Args:
    a: A `Tensor` of type `float32`.
    b: A `Tensor` of type `float32`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `int32`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "TwoFloatInputsIntOutput", name, a, b)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_two_float_inputs_int_output(
          (a, b, name,), None)
      if _result is not NotImplemented:
        return _result
      return two_float_inputs_int_output_eager_fallback(
          a, b, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            two_float_inputs_int_output, (), dict(a=a, b=b, name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_two_float_inputs_int_output(
        (a, b, name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "TwoFloatInputsIntOutput", a=a, b=b, name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          two_float_inputs_int_output, (), dict(a=a, b=b, name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ()
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "TwoFloatInputsIntOutput", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

TwoFloatInputsIntOutput = tf_export("raw_ops.TwoFloatInputsIntOutput")(_ops.to_raw_op(two_float_inputs_int_output))
_dispatcher_for_two_float_inputs_int_output = two_float_inputs_int_output._tf_type_based_dispatcher.Dispatch


def two_float_inputs_int_output_eager_fallback(a: Annotated[Any, _atypes.Float32], b: Annotated[Any, _atypes.Float32], name, ctx) -> Annotated[Any, _atypes.Int32]:
  a = _ops.convert_to_tensor(a, _dtypes.float32)
  b = _ops.convert_to_tensor(b, _dtypes.float32)
  _inputs_flat = [a, b]
  _attrs = None
  _result = _execute.execute(b"TwoFloatInputsIntOutput", 1,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "TwoFloatInputsIntOutput", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

_TwoFloatOutputsOutput = collections.namedtuple(
    "TwoFloatOutputs",
    ["a", "b"])


@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('two_float_outputs')
def two_float_outputs(name=None):
  r"""TODO: add doc.

  Args:
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (a, b).

    a: A `Tensor` of type `float32`.
    b: A `Tensor` of type `float32`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "TwoFloatOutputs", name)
      _result = _TwoFloatOutputsOutput._make(_result)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_two_float_outputs(
          (name,), None)
      if _result is not NotImplemented:
        return _result
      return two_float_outputs_eager_fallback(
          name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            two_float_outputs, (), dict(name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_two_float_outputs(
        (name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "TwoFloatOutputs", name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          two_float_outputs, (), dict(name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ()
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "TwoFloatOutputs", _inputs_flat, _attrs, _result)
  _result = _TwoFloatOutputsOutput._make(_result)
  return _result

TwoFloatOutputs = tf_export("raw_ops.TwoFloatOutputs")(_ops.to_raw_op(two_float_outputs))
_dispatcher_for_two_float_outputs = two_float_outputs._tf_type_based_dispatcher.Dispatch


def two_float_outputs_eager_fallback(name, ctx):
  _inputs_flat = []
  _attrs = None
  _result = _execute.execute(b"TwoFloatOutputs", 2, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "TwoFloatOutputs", _inputs_flat, _attrs, _result)
  _result = _TwoFloatOutputsOutput._make(_result)
  return _result


@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('two_int_inputs')
def two_int_inputs(a: Annotated[Any, _atypes.Int32], b: Annotated[Any, _atypes.Int32], name=None):
  r"""TODO: add doc.

  Args:
    a: A `Tensor` of type `int32`.
    b: A `Tensor` of type `int32`.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "TwoIntInputs", name, a, b)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_two_int_inputs(
          (a, b, name,), None)
      if _result is not NotImplemented:
        return _result
      return two_int_inputs_eager_fallback(
          a, b, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            two_int_inputs, (), dict(a=a, b=b, name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_two_int_inputs(
        (a, b, name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "TwoIntInputs", a=a, b=b, name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          two_int_inputs, (), dict(a=a, b=b, name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  return _op
TwoIntInputs = tf_export("raw_ops.TwoIntInputs")(_ops.to_raw_op(two_int_inputs))
_dispatcher_for_two_int_inputs = two_int_inputs._tf_type_based_dispatcher.Dispatch


def two_int_inputs_eager_fallback(a: Annotated[Any, _atypes.Int32], b: Annotated[Any, _atypes.Int32], name, ctx):
  a = _ops.convert_to_tensor(a, _dtypes.int32)
  b = _ops.convert_to_tensor(b, _dtypes.int32)
  _inputs_flat = [a, b]
  _attrs = None
  _result = _execute.execute(b"TwoIntInputs", 0, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  _result = None
  return _result

_TwoIntOutputsOutput = collections.namedtuple(
    "TwoIntOutputs",
    ["a", "b"])


@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('two_int_outputs')
def two_int_outputs(name=None):
  r"""TODO: add doc.

  Args:
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (a, b).

    a: A `Tensor` of type `int32`.
    b: A `Tensor` of type `int32`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "TwoIntOutputs", name)
      _result = _TwoIntOutputsOutput._make(_result)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_two_int_outputs(
          (name,), None)
      if _result is not NotImplemented:
        return _result
      return two_int_outputs_eager_fallback(
          name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            two_int_outputs, (), dict(name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_two_int_outputs(
        (name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "TwoIntOutputs", name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          two_int_outputs, (), dict(name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ()
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "TwoIntOutputs", _inputs_flat, _attrs, _result)
  _result = _TwoIntOutputsOutput._make(_result)
  return _result

TwoIntOutputs = tf_export("raw_ops.TwoIntOutputs")(_ops.to_raw_op(two_int_outputs))
_dispatcher_for_two_int_outputs = two_int_outputs._tf_type_based_dispatcher.Dispatch


def two_int_outputs_eager_fallback(name, ctx):
  _inputs_flat = []
  _attrs = None
  _result = _execute.execute(b"TwoIntOutputs", 2, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "TwoIntOutputs", _inputs_flat, _attrs, _result)
  _result = _TwoIntOutputsOutput._make(_result)
  return _result


TV_TwoRefsIn_T = TypeVar("TV_TwoRefsIn_T", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float16, _atypes.Float32, _atypes.Float64, _atypes.Float8e4m3fn, _atypes.Float8e5m2, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int4, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.Resource, _atypes.String, _atypes.UInt16, _atypes.UInt32, _atypes.UInt4, _atypes.UInt64, _atypes.UInt8, _atypes.Variant)

@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('two_refs_in')
def two_refs_in(a: Annotated[Any, TV_TwoRefsIn_T], b: Annotated[Any, TV_TwoRefsIn_T], name=None):
  r"""TODO: add doc.

  Args:
    a: A mutable `Tensor`.
    b: A mutable `Tensor`. Must have the same type as `a`.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    raise RuntimeError("two_refs_in op does not support eager execution. Arg 'b' is a ref.")
  else:
    _result = _dispatcher_for_two_refs_in(
        (a, b, name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "TwoRefsIn", a=a, b=b, name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          two_refs_in, (), dict(a=a, b=b, name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  return _op
TwoRefsIn = tf_export("raw_ops.TwoRefsIn")(_ops.to_raw_op(two_refs_in))
_dispatcher_for_two_refs_in = two_refs_in._tf_type_based_dispatcher.Dispatch


def two_refs_in_eager_fallback(a: Annotated[Any, TV_TwoRefsIn_T], b: Annotated[Any, TV_TwoRefsIn_T], name, ctx):
  raise RuntimeError("two_refs_in op does not support eager execution. Arg 'b' is a ref.")

@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('type_list')
def type_list(a, name=None):
  r"""TODO: add doc.

  Args:
    a: A list of `Tensor` objects.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "TypeList", name, a)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_type_list(
          (a, name,), None)
      if _result is not NotImplemented:
        return _result
      return type_list_eager_fallback(
          a, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            type_list, (), dict(a=a, name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_type_list(
        (a, name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "TypeList", a=a, name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          type_list, (), dict(a=a, name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  return _op
TypeList = tf_export("raw_ops.TypeList")(_ops.to_raw_op(type_list))
_dispatcher_for_type_list = type_list._tf_type_based_dispatcher.Dispatch


def type_list_eager_fallback(a, name, ctx):
  _attr_T, a = _execute.convert_to_mixed_eager_tensors(a, ctx)
  _inputs_flat = list(a)
  _attrs = ("T", _attr_T)
  _result = _execute.execute(b"TypeList", 0, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  _result = None
  return _result


@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('type_list_restrict')
def type_list_restrict(a, name=None):
  r"""TODO: add doc.

  Args:
    a: A list of `Tensor` objects with types from: `string`, `bool`.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "TypeListRestrict", name, a)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_type_list_restrict(
          (a, name,), None)
      if _result is not NotImplemented:
        return _result
      return type_list_restrict_eager_fallback(
          a, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            type_list_restrict, (), dict(a=a, name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_type_list_restrict(
        (a, name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "TypeListRestrict", a=a, name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          type_list_restrict, (), dict(a=a, name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  return _op
TypeListRestrict = tf_export("raw_ops.TypeListRestrict")(_ops.to_raw_op(type_list_restrict))
_dispatcher_for_type_list_restrict = type_list_restrict._tf_type_based_dispatcher.Dispatch


def type_list_restrict_eager_fallback(a, name, ctx):
  _attr_T, a = _execute.convert_to_mixed_eager_tensors(a, ctx)
  _inputs_flat = list(a)
  _attrs = ("T", _attr_T)
  _result = _execute.execute(b"TypeListRestrict", 0, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  _result = None
  return _result


@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('type_list_twice')
def type_list_twice(a, b, name=None):
  r"""TODO: add doc.

  Args:
    a: A list of `Tensor` objects.
    b: A list of `Tensor` objects. Must have the same type as `a`.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "TypeListTwice", name, a, b)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_type_list_twice(
          (a, b, name,), None)
      if _result is not NotImplemented:
        return _result
      return type_list_twice_eager_fallback(
          a, b, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            type_list_twice, (), dict(a=a, b=b, name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_type_list_twice(
        (a, b, name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "TypeListTwice", a=a, b=b, name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          type_list_twice, (), dict(a=a, b=b, name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  return _op
TypeListTwice = tf_export("raw_ops.TypeListTwice")(_ops.to_raw_op(type_list_twice))
_dispatcher_for_type_list_twice = type_list_twice._tf_type_based_dispatcher.Dispatch


def type_list_twice_eager_fallback(a, b, name, ctx):
  _attr_T, (a, b) = _execute.args_to_mixed_eager_tensors((a, b), ctx)
  _inputs_flat = list(a) + list(b)
  _attrs = ("T", _attr_T)
  _result = _execute.execute(b"TypeListTwice", 0, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  _result = None
  return _result


TV_Unary_T = TypeVar("TV_Unary_T", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float16, _atypes.Float32, _atypes.Float64, _atypes.Float8e4m3fn, _atypes.Float8e5m2, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int4, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.Resource, _atypes.String, _atypes.UInt16, _atypes.UInt32, _atypes.UInt4, _atypes.UInt64, _atypes.UInt8, _atypes.Variant)

@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('unary')
def unary(a: Annotated[Any, TV_Unary_T], name=None) -> Annotated[Any, TV_Unary_T]:
  r"""TODO: add doc.

  Args:
    a: A `Tensor`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `a`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "Unary", name, a)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_unary(
          (a, name,), None)
      if _result is not NotImplemented:
        return _result
      return unary_eager_fallback(
          a, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            unary, (), dict(a=a, name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_unary(
        (a, name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "Unary", a=a, name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          unary, (), dict(a=a, name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "Unary", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

Unary = tf_export("raw_ops.Unary")(_ops.to_raw_op(unary))
_dispatcher_for_unary = unary._tf_type_based_dispatcher.Dispatch


def unary_eager_fallback(a: Annotated[Any, TV_Unary_T], name, ctx) -> Annotated[Any, TV_Unary_T]:
  _attr_T, (a,) = _execute.args_to_matching_eager([a], ctx, [])
  _inputs_flat = [a]
  _attrs = ("T", _attr_T)
  _result = _execute.execute(b"Unary", 1, inputs=_inputs_flat, attrs=_attrs,
                             ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "Unary", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

