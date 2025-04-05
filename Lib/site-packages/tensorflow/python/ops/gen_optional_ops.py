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

def optional_from_value(components, name=None) -> Annotated[Any, _atypes.Variant]:
  r"""Constructs an Optional variant from a tuple of tensors.

  Args:
    components: A list of `Tensor` objects.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `variant`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "OptionalFromValue", name, components)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return optional_from_value_eager_fallback(
          components, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "OptionalFromValue", components=components, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("Toutput_types", _op.get_attr("Toutput_types"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "OptionalFromValue", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

OptionalFromValue = tf_export("raw_ops.OptionalFromValue")(_ops.to_raw_op(optional_from_value))


def optional_from_value_eager_fallback(components, name, ctx) -> Annotated[Any, _atypes.Variant]:
  _attr_Toutput_types, components = _execute.convert_to_mixed_eager_tensors(components, ctx)
  _inputs_flat = list(components)
  _attrs = ("Toutput_types", _attr_Toutput_types)
  _result = _execute.execute(b"OptionalFromValue", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "OptionalFromValue", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


def optional_get_value(optional: Annotated[Any, _atypes.Variant], output_types, output_shapes, name=None):
  r"""Returns the value stored in an Optional variant or raises an error if none exists.

  Args:
    optional: A `Tensor` of type `variant`.
    output_types: A list of `tf.DTypes` that has length `>= 1`.
    output_shapes: A list of shapes (each a `tf.TensorShape` or list of `ints`) that has length `>= 1`.
    name: A name for the operation (optional).

  Returns:
    A list of `Tensor` objects of type `output_types`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "OptionalGetValue", name, optional, "output_types",
        output_types, "output_shapes", output_shapes)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return optional_get_value_eager_fallback(
          optional, output_types=output_types, output_shapes=output_shapes,
          name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if not isinstance(output_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_types' argument to "
        "'optional_get_value' Op, not %r." % output_types)
  output_types = [_execute.make_type(_t, "output_types") for _t in output_types]
  if not isinstance(output_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_shapes' argument to "
        "'optional_get_value' Op, not %r." % output_shapes)
  output_shapes = [_execute.make_shape(_s, "output_shapes") for _s in output_shapes]
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "OptionalGetValue", optional=optional, output_types=output_types,
                            output_shapes=output_shapes, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("output_types", _op.get_attr("output_types"), "output_shapes",
              _op.get_attr("output_shapes"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "OptionalGetValue", _inputs_flat, _attrs, _result)
  return _result

OptionalGetValue = tf_export("raw_ops.OptionalGetValue")(_ops.to_raw_op(optional_get_value))


def optional_get_value_eager_fallback(optional: Annotated[Any, _atypes.Variant], output_types, output_shapes, name, ctx):
  if not isinstance(output_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_types' argument to "
        "'optional_get_value' Op, not %r." % output_types)
  output_types = [_execute.make_type(_t, "output_types") for _t in output_types]
  if not isinstance(output_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_shapes' argument to "
        "'optional_get_value' Op, not %r." % output_shapes)
  output_shapes = [_execute.make_shape(_s, "output_shapes") for _s in output_shapes]
  optional = _ops.convert_to_tensor(optional, _dtypes.variant)
  _inputs_flat = [optional]
  _attrs = ("output_types", output_types, "output_shapes", output_shapes)
  _result = _execute.execute(b"OptionalGetValue", len(output_types),
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "OptionalGetValue", _inputs_flat, _attrs, _result)
  return _result


def optional_has_value(optional: Annotated[Any, _atypes.Variant], name=None) -> Annotated[Any, _atypes.Bool]:
  r"""Returns true if and only if the given Optional variant has a value.

  Args:
    optional: A `Tensor` of type `variant`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `bool`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "OptionalHasValue", name, optional)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return optional_has_value_eager_fallback(
          optional, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "OptionalHasValue", optional=optional, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ()
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "OptionalHasValue", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

OptionalHasValue = tf_export("raw_ops.OptionalHasValue")(_ops.to_raw_op(optional_has_value))


def optional_has_value_eager_fallback(optional: Annotated[Any, _atypes.Variant], name, ctx) -> Annotated[Any, _atypes.Bool]:
  optional = _ops.convert_to_tensor(optional, _dtypes.variant)
  _inputs_flat = [optional]
  _attrs = None
  _result = _execute.execute(b"OptionalHasValue", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "OptionalHasValue", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


def optional_none(name=None) -> Annotated[Any, _atypes.Variant]:
  r"""Creates an Optional variant with no value.

  Args:
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `variant`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "OptionalNone", name)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return optional_none_eager_fallback(
          name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "OptionalNone", name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ()
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "OptionalNone", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

OptionalNone = tf_export("raw_ops.OptionalNone")(_ops.to_raw_op(optional_none))


def optional_none_eager_fallback(name, ctx) -> Annotated[Any, _atypes.Variant]:
  _inputs_flat = []
  _attrs = None
  _result = _execute.execute(b"OptionalNone", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "OptionalNone", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

