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
@tf_export('mlir_passthrough_op')
def mlir_passthrough_op(inputs, mlir_module: str, Toutputs, name=None):
  r"""TODO: add doc.

  Args:
    inputs: A list of `Tensor` objects.
    mlir_module: A `string`.
    Toutputs: A list of `tf.DTypes`.
    name: A name for the operation (optional).

  Returns:
    A list of `Tensor` objects of type `Toutputs`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "MlirPassthroughOp", name, inputs, "mlir_module", mlir_module,
        "Toutputs", Toutputs)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_mlir_passthrough_op(
          (inputs, mlir_module, Toutputs, name,), None)
      if _result is not NotImplemented:
        return _result
      return mlir_passthrough_op_eager_fallback(
          inputs, mlir_module=mlir_module, Toutputs=Toutputs, name=name,
          ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            mlir_passthrough_op, (), dict(inputs=inputs,
                                          mlir_module=mlir_module,
                                          Toutputs=Toutputs, name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_mlir_passthrough_op(
        (inputs, mlir_module, Toutputs, name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  mlir_module = _execute.make_str(mlir_module, "mlir_module")
  if not isinstance(Toutputs, (list, tuple)):
    raise TypeError(
        "Expected list for 'Toutputs' argument to "
        "'mlir_passthrough_op' Op, not %r." % Toutputs)
  Toutputs = [_execute.make_type(_t, "Toutputs") for _t in Toutputs]
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "MlirPassthroughOp", inputs=inputs, mlir_module=mlir_module,
                             Toutputs=Toutputs, name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          mlir_passthrough_op, (), dict(inputs=inputs,
                                        mlir_module=mlir_module,
                                        Toutputs=Toutputs, name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("mlir_module", _op.get_attr("mlir_module"), "Tinputs",
              _op.get_attr("Tinputs"), "Toutputs", _op.get_attr("Toutputs"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "MlirPassthroughOp", _inputs_flat, _attrs, _result)
  return _result

MlirPassthroughOp = tf_export("raw_ops.MlirPassthroughOp")(_ops.to_raw_op(mlir_passthrough_op))
_dispatcher_for_mlir_passthrough_op = mlir_passthrough_op._tf_type_based_dispatcher.Dispatch


def mlir_passthrough_op_eager_fallback(inputs, mlir_module: str, Toutputs, name, ctx):
  mlir_module = _execute.make_str(mlir_module, "mlir_module")
  if not isinstance(Toutputs, (list, tuple)):
    raise TypeError(
        "Expected list for 'Toutputs' argument to "
        "'mlir_passthrough_op' Op, not %r." % Toutputs)
  Toutputs = [_execute.make_type(_t, "Toutputs") for _t in Toutputs]
  _attr_Tinputs, inputs = _execute.convert_to_mixed_eager_tensors(inputs, ctx)
  _inputs_flat = list(inputs)
  _attrs = ("mlir_module", mlir_module, "Tinputs", _attr_Tinputs, "Toutputs",
  Toutputs)
  _result = _execute.execute(b"MlirPassthroughOp", len(Toutputs),
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "MlirPassthroughOp", _inputs_flat, _attrs, _result)
  return _result

