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

def composite_tensor_variant_from_components(components, metadata: str, name=None) -> Annotated[Any, _atypes.Variant]:
  r"""Encodes an `ExtensionType` value into a `variant` scalar Tensor.

  Returns a scalar variant tensor containing a single `CompositeTensorVariant`
  with the specified Tensor components and TypeSpec.

  Args:
    components: A list of `Tensor` objects.
      The component tensors for the extension type value.
    metadata: A `string`.
      String serialization for the TypeSpec.  (Note: the encoding for the TypeSpec
      may change in future versions of TensorFlow.)
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `variant`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "CompositeTensorVariantFromComponents", name, components,
        "metadata", metadata)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return composite_tensor_variant_from_components_eager_fallback(
          components, metadata=metadata, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  metadata = _execute.make_str(metadata, "metadata")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "CompositeTensorVariantFromComponents", components=components,
                                                metadata=metadata, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("metadata", _op.get_attr("metadata"), "Tcomponents",
              _op.get_attr("Tcomponents"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "CompositeTensorVariantFromComponents", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

CompositeTensorVariantFromComponents = tf_export("raw_ops.CompositeTensorVariantFromComponents")(_ops.to_raw_op(composite_tensor_variant_from_components))


def composite_tensor_variant_from_components_eager_fallback(components, metadata: str, name, ctx) -> Annotated[Any, _atypes.Variant]:
  metadata = _execute.make_str(metadata, "metadata")
  _attr_Tcomponents, components = _execute.convert_to_mixed_eager_tensors(components, ctx)
  _inputs_flat = list(components)
  _attrs = ("metadata", metadata, "Tcomponents", _attr_Tcomponents)
  _result = _execute.execute(b"CompositeTensorVariantFromComponents", 1,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "CompositeTensorVariantFromComponents", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


def composite_tensor_variant_to_components(encoded: Annotated[Any, _atypes.Variant], metadata: str, Tcomponents, name=None):
  r"""Decodes a `variant` scalar Tensor into an `ExtensionType` value.

  Returns the Tensor components encoded in a `CompositeTensorVariant`.

  Raises an error if `type_spec_proto` doesn't match the TypeSpec
  in `encoded`.

  Args:
    encoded: A `Tensor` of type `variant`.
      A scalar `variant` Tensor containing an encoded ExtensionType value.
    metadata: A `string`.
      String serialization for the TypeSpec.  Must be compatible with the
      `TypeSpec` contained in `encoded`.  (Note: the encoding for the TypeSpec
      may change in future versions of TensorFlow.)
    Tcomponents: A list of `tf.DTypes`. Expected dtypes for components.
    name: A name for the operation (optional).

  Returns:
    A list of `Tensor` objects of type `Tcomponents`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "CompositeTensorVariantToComponents", name, encoded, "metadata",
        metadata, "Tcomponents", Tcomponents)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return composite_tensor_variant_to_components_eager_fallback(
          encoded, metadata=metadata, Tcomponents=Tcomponents, name=name,
          ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  metadata = _execute.make_str(metadata, "metadata")
  if not isinstance(Tcomponents, (list, tuple)):
    raise TypeError(
        "Expected list for 'Tcomponents' argument to "
        "'composite_tensor_variant_to_components' Op, not %r." % Tcomponents)
  Tcomponents = [_execute.make_type(_t, "Tcomponents") for _t in Tcomponents]
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "CompositeTensorVariantToComponents", encoded=encoded,
                                              metadata=metadata,
                                              Tcomponents=Tcomponents,
                                              name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("metadata", _op.get_attr("metadata"), "Tcomponents",
              _op.get_attr("Tcomponents"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "CompositeTensorVariantToComponents", _inputs_flat, _attrs, _result)
  return _result

CompositeTensorVariantToComponents = tf_export("raw_ops.CompositeTensorVariantToComponents")(_ops.to_raw_op(composite_tensor_variant_to_components))


def composite_tensor_variant_to_components_eager_fallback(encoded: Annotated[Any, _atypes.Variant], metadata: str, Tcomponents, name, ctx):
  metadata = _execute.make_str(metadata, "metadata")
  if not isinstance(Tcomponents, (list, tuple)):
    raise TypeError(
        "Expected list for 'Tcomponents' argument to "
        "'composite_tensor_variant_to_components' Op, not %r." % Tcomponents)
  Tcomponents = [_execute.make_type(_t, "Tcomponents") for _t in Tcomponents]
  encoded = _ops.convert_to_tensor(encoded, _dtypes.variant)
  _inputs_flat = [encoded]
  _attrs = ("metadata", metadata, "Tcomponents", Tcomponents)
  _result = _execute.execute(b"CompositeTensorVariantToComponents",
                             len(Tcomponents), inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "CompositeTensorVariantToComponents", _inputs_flat, _attrs, _result)
  return _result

