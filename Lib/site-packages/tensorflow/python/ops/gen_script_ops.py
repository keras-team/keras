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

def eager_py_func(input, token: str, Tout, is_async:bool=False, name=None):
  r"""Eagerly executes a python function to compute func(input)->output. The

  semantics of the input, output, and attributes are the same as those for
  PyFunc.

  Args:
    input: A list of `Tensor` objects.
    token: A `string`.
    Tout: A list of `tf.DTypes`.
    is_async: An optional `bool`. Defaults to `False`.
    name: A name for the operation (optional).

  Returns:
    A list of `Tensor` objects of type `Tout`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "EagerPyFunc", name, input, "token", token, "is_async",
        is_async, "Tout", Tout)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return eager_py_func_eager_fallback(
          input, token=token, is_async=is_async, Tout=Tout, name=name,
          ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  token = _execute.make_str(token, "token")
  if not isinstance(Tout, (list, tuple)):
    raise TypeError(
        "Expected list for 'Tout' argument to "
        "'eager_py_func' Op, not %r." % Tout)
  Tout = [_execute.make_type(_t, "Tout") for _t in Tout]
  if is_async is None:
    is_async = False
  is_async = _execute.make_bool(is_async, "is_async")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "EagerPyFunc", input=input, token=token, Tout=Tout, is_async=is_async,
                       name=name)
  _result = _outputs[:]
  if not _result:
    return _op
  if _execute.must_record_gradient():
    _attrs = ("token", _op.get_attr("token"), "is_async",
              _op._get_attr_bool("is_async"), "Tin", _op.get_attr("Tin"),
              "Tout", _op.get_attr("Tout"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "EagerPyFunc", _inputs_flat, _attrs, _result)
  return _result

EagerPyFunc = tf_export("raw_ops.EagerPyFunc")(_ops.to_raw_op(eager_py_func))


def eager_py_func_eager_fallback(input, token: str, Tout, is_async: bool, name, ctx):
  token = _execute.make_str(token, "token")
  if not isinstance(Tout, (list, tuple)):
    raise TypeError(
        "Expected list for 'Tout' argument to "
        "'eager_py_func' Op, not %r." % Tout)
  Tout = [_execute.make_type(_t, "Tout") for _t in Tout]
  if is_async is None:
    is_async = False
  is_async = _execute.make_bool(is_async, "is_async")
  _attr_Tin, input = _execute.convert_to_mixed_eager_tensors(input, ctx)
  _inputs_flat = list(input)
  _attrs = ("token", token, "is_async", is_async, "Tin", _attr_Tin, "Tout",
  Tout)
  _result = _execute.execute(b"EagerPyFunc", len(Tout), inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "EagerPyFunc", _inputs_flat, _attrs, _result)
  return _result


def py_func(input, token: str, Tout, name=None):
  r"""Invokes a python function to compute func(input)->output.

  This operation is considered stateful. For a stateless version, see
  PyFuncStateless.

  Args:
    input: A list of `Tensor` objects.
      List of Tensors that will provide input to the Op.
    token: A `string`.
      A token representing a registered python function in this address space.
    Tout: A list of `tf.DTypes`. Data types of the outputs from the op.
      The length of the list specifies the number of outputs.
    name: A name for the operation (optional).

  Returns:
    A list of `Tensor` objects of type `Tout`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "PyFunc", name, input, "token", token, "Tout", Tout)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return py_func_eager_fallback(
          input, token=token, Tout=Tout, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  token = _execute.make_str(token, "token")
  if not isinstance(Tout, (list, tuple)):
    raise TypeError(
        "Expected list for 'Tout' argument to "
        "'py_func' Op, not %r." % Tout)
  Tout = [_execute.make_type(_t, "Tout") for _t in Tout]
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "PyFunc", input=input, token=token, Tout=Tout, name=name)
  _result = _outputs[:]
  if not _result:
    return _op
  if _execute.must_record_gradient():
    _attrs = ("token", _op.get_attr("token"), "Tin", _op.get_attr("Tin"),
              "Tout", _op.get_attr("Tout"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "PyFunc", _inputs_flat, _attrs, _result)
  return _result

PyFunc = tf_export("raw_ops.PyFunc")(_ops.to_raw_op(py_func))


def py_func_eager_fallback(input, token: str, Tout, name, ctx):
  token = _execute.make_str(token, "token")
  if not isinstance(Tout, (list, tuple)):
    raise TypeError(
        "Expected list for 'Tout' argument to "
        "'py_func' Op, not %r." % Tout)
  Tout = [_execute.make_type(_t, "Tout") for _t in Tout]
  _attr_Tin, input = _execute.convert_to_mixed_eager_tensors(input, ctx)
  _inputs_flat = list(input)
  _attrs = ("token", token, "Tin", _attr_Tin, "Tout", Tout)
  _result = _execute.execute(b"PyFunc", len(Tout), inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "PyFunc", _inputs_flat, _attrs, _result)
  return _result


def py_func_stateless(input, token: str, Tout, name=None):
  r"""A stateless version of PyFunc.

  Args:
    input: A list of `Tensor` objects.
    token: A `string`.
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
        _ctx, "PyFuncStateless", name, input, "token", token, "Tout", Tout)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return py_func_stateless_eager_fallback(
          input, token=token, Tout=Tout, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  token = _execute.make_str(token, "token")
  if not isinstance(Tout, (list, tuple)):
    raise TypeError(
        "Expected list for 'Tout' argument to "
        "'py_func_stateless' Op, not %r." % Tout)
  Tout = [_execute.make_type(_t, "Tout") for _t in Tout]
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "PyFuncStateless", input=input, token=token, Tout=Tout, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("token", _op.get_attr("token"), "Tin", _op.get_attr("Tin"),
              "Tout", _op.get_attr("Tout"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "PyFuncStateless", _inputs_flat, _attrs, _result)
  return _result

PyFuncStateless = tf_export("raw_ops.PyFuncStateless")(_ops.to_raw_op(py_func_stateless))


def py_func_stateless_eager_fallback(input, token: str, Tout, name, ctx):
  token = _execute.make_str(token, "token")
  if not isinstance(Tout, (list, tuple)):
    raise TypeError(
        "Expected list for 'Tout' argument to "
        "'py_func_stateless' Op, not %r." % Tout)
  Tout = [_execute.make_type(_t, "Tout") for _t in Tout]
  _attr_Tin, input = _execute.convert_to_mixed_eager_tensors(input, ctx)
  _inputs_flat = list(input)
  _attrs = ("token", token, "Tin", _attr_Tin, "Tout", Tout)
  _result = _execute.execute(b"PyFuncStateless", len(Tout),
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "PyFuncStateless", _inputs_flat, _attrs, _result)
  return _result

