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
@tf_export('io.encode_proto')
def encode_proto(sizes: Annotated[Any, _atypes.Int32], values, field_names, message_type: str, descriptor_source:str="local://", name=None) -> Annotated[Any, _atypes.String]:
  r"""The op serializes protobuf messages provided in the input tensors.

  The types of the tensors in `values` must match the schema for the fields
  specified in `field_names`. All the tensors in `values` must have a common
  shape prefix, *batch_shape*.

  The `sizes` tensor specifies repeat counts for each field.  The repeat count
  (last dimension) of a each tensor in `values` must be greater than or equal
  to corresponding repeat count in `sizes`.

  A `message_type` name must be provided to give context for the field names.
  The actual message descriptor can be looked up either in the linked-in
  descriptor pool or a filename provided by the caller using the
  `descriptor_source` attribute.

  For the most part, the mapping between Proto field types and TensorFlow dtypes
  is straightforward. However, there are a few special cases:

  - A proto field that contains a submessage or group can only be converted
  to `DT_STRING` (the serialized submessage). This is to reduce the complexity
  of the API. The resulting string can be used as input to another instance of
  the decode_proto op.

  - TensorFlow lacks support for unsigned integers. The ops represent uint64
  types as a `DT_INT64` with the same twos-complement bit pattern (the obvious
  way). Unsigned int32 values can be represented exactly by specifying type
  `DT_INT64`, or using twos-complement if the caller specifies `DT_INT32` in
  the `output_types` attribute.

  The `descriptor_source` attribute selects the source of protocol
  descriptors to consult when looking up `message_type`. This may be:

  - An empty string  or "local://", in which case protocol descriptors are
  created for C++ (not Python) proto definitions linked to the binary.

  - A file, in which case protocol descriptors are created from the file,
  which is expected to contain a `FileDescriptorSet` serialized as a string.
  NOTE: You can build a `descriptor_source` file using the `--descriptor_set_out`
  and `--include_imports` options to the protocol compiler `protoc`.

  - A "bytes://<bytes>", in which protocol descriptors are created from `<bytes>`,
  which is expected to be a `FileDescriptorSet` serialized as a string.

  Args:
    sizes: A `Tensor` of type `int32`.
      Tensor of int32 with shape `[batch_shape, len(field_names)]`.
    values: A list of `Tensor` objects.
      List of tensors containing values for the corresponding field.
    field_names: A list of `strings`.
      List of strings containing proto field names.
    message_type: A `string`. Name of the proto message type to decode.
    descriptor_source: An optional `string`. Defaults to `"local://"`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `string`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "EncodeProto", name, sizes, values, "field_names", field_names,
        "message_type", message_type, "descriptor_source", descriptor_source)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_encode_proto(
          (sizes, values, field_names, message_type, descriptor_source,
          name,), None)
      if _result is not NotImplemented:
        return _result
      return encode_proto_eager_fallback(
          sizes, values, field_names=field_names, message_type=message_type,
          descriptor_source=descriptor_source, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            encode_proto, (), dict(sizes=sizes, values=values,
                                   field_names=field_names,
                                   message_type=message_type,
                                   descriptor_source=descriptor_source,
                                   name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_encode_proto(
        (sizes, values, field_names, message_type, descriptor_source, name,),
        None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  if not isinstance(field_names, (list, tuple)):
    raise TypeError(
        "Expected list for 'field_names' argument to "
        "'encode_proto' Op, not %r." % field_names)
  field_names = [_execute.make_str(_s, "field_names") for _s in field_names]
  message_type = _execute.make_str(message_type, "message_type")
  if descriptor_source is None:
    descriptor_source = "local://"
  descriptor_source = _execute.make_str(descriptor_source, "descriptor_source")
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "EncodeProto", sizes=sizes, values=values, field_names=field_names,
                       message_type=message_type,
                       descriptor_source=descriptor_source, name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          encode_proto, (), dict(sizes=sizes, values=values,
                                 field_names=field_names,
                                 message_type=message_type,
                                 descriptor_source=descriptor_source,
                                 name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("field_names", _op.get_attr("field_names"), "message_type",
              _op.get_attr("message_type"), "descriptor_source",
              _op.get_attr("descriptor_source"), "Tinput_types",
              _op.get_attr("Tinput_types"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "EncodeProto", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

EncodeProto = tf_export("raw_ops.EncodeProto")(_ops.to_raw_op(encode_proto))
_dispatcher_for_encode_proto = encode_proto._tf_type_based_dispatcher.Dispatch


def encode_proto_eager_fallback(sizes: Annotated[Any, _atypes.Int32], values, field_names, message_type: str, descriptor_source: str, name, ctx) -> Annotated[Any, _atypes.String]:
  if not isinstance(field_names, (list, tuple)):
    raise TypeError(
        "Expected list for 'field_names' argument to "
        "'encode_proto' Op, not %r." % field_names)
  field_names = [_execute.make_str(_s, "field_names") for _s in field_names]
  message_type = _execute.make_str(message_type, "message_type")
  if descriptor_source is None:
    descriptor_source = "local://"
  descriptor_source = _execute.make_str(descriptor_source, "descriptor_source")
  _attr_Tinput_types, values = _execute.convert_to_mixed_eager_tensors(values, ctx)
  sizes = _ops.convert_to_tensor(sizes, _dtypes.int32)
  _inputs_flat = [sizes] + list(values)
  _attrs = ("field_names", field_names, "message_type", message_type,
  "descriptor_source", descriptor_source, "Tinput_types", _attr_Tinput_types)
  _result = _execute.execute(b"EncodeProto", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "EncodeProto", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

