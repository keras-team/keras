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

TV_AsString_T = TypeVar("TV_AsString_T", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float32, _atypes.Float64, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int64, _atypes.Int8, _atypes.String, _atypes.UInt16, _atypes.UInt32, _atypes.UInt64, _atypes.UInt8, _atypes.Variant)

@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('strings.as_string', 'as_string', v1=['dtypes.as_string', 'strings.as_string', 'as_string'])
@deprecated_endpoints('dtypes.as_string')
def as_string(input: Annotated[Any, TV_AsString_T], precision:int=-1, scientific:bool=False, shortest:bool=False, width:int=-1, fill:str="", name=None) -> Annotated[Any, _atypes.String]:
  r"""Converts each entry in the given tensor to strings.

  Supports many numeric types and boolean.

  For Unicode, see the
  [https://www.tensorflow.org/tutorials/representation/unicode](Working with Unicode text)
  tutorial.

  Examples:

  >>> tf.strings.as_string([3, 2])
  <tf.Tensor: shape=(2,), dtype=string, numpy=array([b'3', b'2'], dtype=object)>
  >>> tf.strings.as_string([3.1415926, 2.71828], precision=2).numpy()
  array([b'3.14', b'2.72'], dtype=object)

  Args:
    input: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`, `int64`, `bfloat16`, `uint16`, `half`, `uint32`, `uint64`, `complex64`, `complex128`, `bool`, `variant`, `string`.
    precision: An optional `int`. Defaults to `-1`.
      The post-decimal precision to use for floating point numbers.
      Only used if precision > -1.
    scientific: An optional `bool`. Defaults to `False`.
      Use scientific notation for floating point numbers.
    shortest: An optional `bool`. Defaults to `False`.
      Use shortest representation (either scientific or standard) for
      floating point numbers.
    width: An optional `int`. Defaults to `-1`.
      Pad pre-decimal numbers to this width.
      Applies to both floating point and integer numbers.
      Only used if width > -1.
    fill: An optional `string`. Defaults to `""`.
      The value to pad if width > -1.  If empty, pads with spaces.
      Another typical value is '0'.  String cannot be longer than 1 character.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `string`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "AsString", name, input, "precision", precision, "scientific",
        scientific, "shortest", shortest, "width", width, "fill", fill)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_as_string(
          (input, precision, scientific, shortest, width, fill, name,), None)
      if _result is not NotImplemented:
        return _result
      return as_string_eager_fallback(
          input, precision=precision, scientific=scientific,
          shortest=shortest, width=width, fill=fill, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            as_string, (), dict(input=input, precision=precision,
                                scientific=scientific, shortest=shortest,
                                width=width, fill=fill, name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_as_string(
        (input, precision, scientific, shortest, width, fill, name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  if precision is None:
    precision = -1
  precision = _execute.make_int(precision, "precision")
  if scientific is None:
    scientific = False
  scientific = _execute.make_bool(scientific, "scientific")
  if shortest is None:
    shortest = False
  shortest = _execute.make_bool(shortest, "shortest")
  if width is None:
    width = -1
  width = _execute.make_int(width, "width")
  if fill is None:
    fill = ""
  fill = _execute.make_str(fill, "fill")
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "AsString", input=input, precision=precision, scientific=scientific,
                    shortest=shortest, width=width, fill=fill, name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          as_string, (), dict(input=input, precision=precision,
                              scientific=scientific, shortest=shortest,
                              width=width, fill=fill, name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"), "precision",
              _op._get_attr_int("precision"), "scientific",
              _op._get_attr_bool("scientific"), "shortest",
              _op._get_attr_bool("shortest"), "width",
              _op._get_attr_int("width"), "fill", _op.get_attr("fill"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "AsString", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

AsString = tf_export("raw_ops.AsString")(_ops.to_raw_op(as_string))
_dispatcher_for_as_string = as_string._tf_type_based_dispatcher.Dispatch


def as_string_eager_fallback(input: Annotated[Any, TV_AsString_T], precision: int, scientific: bool, shortest: bool, width: int, fill: str, name, ctx) -> Annotated[Any, _atypes.String]:
  if precision is None:
    precision = -1
  precision = _execute.make_int(precision, "precision")
  if scientific is None:
    scientific = False
  scientific = _execute.make_bool(scientific, "scientific")
  if shortest is None:
    shortest = False
  shortest = _execute.make_bool(shortest, "shortest")
  if width is None:
    width = -1
  width = _execute.make_int(width, "width")
  if fill is None:
    fill = ""
  fill = _execute.make_str(fill, "fill")
  _attr_T, (input,) = _execute.args_to_matching_eager([input], ctx, [_dtypes.float32, _dtypes.float64, _dtypes.int32, _dtypes.uint8, _dtypes.int16, _dtypes.int8, _dtypes.int64, _dtypes.bfloat16, _dtypes.uint16, _dtypes.half, _dtypes.uint32, _dtypes.uint64, _dtypes.complex64, _dtypes.complex128, _dtypes.bool, _dtypes.variant, _dtypes.string, ])
  _inputs_flat = [input]
  _attrs = ("T", _attr_T, "precision", precision, "scientific", scientific,
  "shortest", shortest, "width", width, "fill", fill)
  _result = _execute.execute(b"AsString", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "AsString", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('io.decode_base64', v1=['io.decode_base64', 'decode_base64'])
@deprecated_endpoints('decode_base64')
def decode_base64(input: Annotated[Any, _atypes.String], name=None) -> Annotated[Any, _atypes.String]:
  r"""Decode web-safe base64-encoded strings.

  Input may or may not have padding at the end. See
  [EncodeBase64](https://www.tensorflow.org/api_docs/python/tf/io/encode_base64)
  for padding. Web-safe means that input must use - and _ instead of + and /.

  Args:
    input: A `Tensor` of type `string`. Base64 strings to decode.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `string`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "DecodeBase64", name, input)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_decode_base64(
          (input, name,), None)
      if _result is not NotImplemented:
        return _result
      return decode_base64_eager_fallback(
          input, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            decode_base64, (), dict(input=input, name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_decode_base64(
        (input, name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "DecodeBase64", input=input, name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          decode_base64, (), dict(input=input, name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ()
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "DecodeBase64", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

DecodeBase64 = tf_export("raw_ops.DecodeBase64")(_ops.to_raw_op(decode_base64))
_dispatcher_for_decode_base64 = decode_base64._tf_type_based_dispatcher.Dispatch


def decode_base64_eager_fallback(input: Annotated[Any, _atypes.String], name, ctx) -> Annotated[Any, _atypes.String]:
  input = _ops.convert_to_tensor(input, _dtypes.string)
  _inputs_flat = [input]
  _attrs = None
  _result = _execute.execute(b"DecodeBase64", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "DecodeBase64", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('io.encode_base64', v1=['io.encode_base64', 'encode_base64'])
@deprecated_endpoints('encode_base64')
def encode_base64(input: Annotated[Any, _atypes.String], pad:bool=False, name=None) -> Annotated[Any, _atypes.String]:
  r"""Encode strings into web-safe base64 format.

  Refer to [this article](https://en.wikipedia.org/wiki/Base64) for more information on
  base64 format. Base64 strings may have padding with '=' at the
  end so that the encoded has length multiple of 4. See Padding section of the
  link above.

  Web-safe means that the encoder uses - and _ instead of + and /.

  Args:
    input: A `Tensor` of type `string`. Strings to be encoded.
    pad: An optional `bool`. Defaults to `False`.
      Bool whether padding is applied at the ends.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `string`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "EncodeBase64", name, input, "pad", pad)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_encode_base64(
          (input, pad, name,), None)
      if _result is not NotImplemented:
        return _result
      return encode_base64_eager_fallback(
          input, pad=pad, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            encode_base64, (), dict(input=input, pad=pad, name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_encode_base64(
        (input, pad, name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  if pad is None:
    pad = False
  pad = _execute.make_bool(pad, "pad")
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "EncodeBase64", input=input, pad=pad, name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          encode_base64, (), dict(input=input, pad=pad, name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("pad", _op._get_attr_bool("pad"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "EncodeBase64", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

EncodeBase64 = tf_export("raw_ops.EncodeBase64")(_ops.to_raw_op(encode_base64))
_dispatcher_for_encode_base64 = encode_base64._tf_type_based_dispatcher.Dispatch


def encode_base64_eager_fallback(input: Annotated[Any, _atypes.String], pad: bool, name, ctx) -> Annotated[Any, _atypes.String]:
  if pad is None:
    pad = False
  pad = _execute.make_bool(pad, "pad")
  input = _ops.convert_to_tensor(input, _dtypes.string)
  _inputs_flat = [input]
  _attrs = ("pad", pad)
  _result = _execute.execute(b"EncodeBase64", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "EncodeBase64", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


def reduce_join(inputs: Annotated[Any, _atypes.String], reduction_indices: Annotated[Any, _atypes.Int32], keep_dims:bool=False, separator:str="", name=None) -> Annotated[Any, _atypes.String]:
  r"""Joins a string Tensor across the given dimensions.

  Computes the string join across dimensions in the given string Tensor of shape
  `[\\(d_0, d_1, ..., d_{n-1}\\)]`.  Returns a new Tensor created by joining the input
  strings with the given separator (default: empty string).  Negative indices are
  counted backwards from the end, with `-1` being equivalent to `n - 1`.  If
  indices are not specified, joins across all dimensions beginning from `n - 1`
  through `0`.

  For example:

  ```python
  # tensor `a` is [["a", "b"], ["c", "d"]]
  tf.reduce_join(a, 0) ==> ["ac", "bd"]
  tf.reduce_join(a, 1) ==> ["ab", "cd"]
  tf.reduce_join(a, -2) = tf.reduce_join(a, 0) ==> ["ac", "bd"]
  tf.reduce_join(a, -1) = tf.reduce_join(a, 1) ==> ["ab", "cd"]
  tf.reduce_join(a, 0, keep_dims=True) ==> [["ac", "bd"]]
  tf.reduce_join(a, 1, keep_dims=True) ==> [["ab"], ["cd"]]
  tf.reduce_join(a, 0, separator=".") ==> ["a.c", "b.d"]
  tf.reduce_join(a, [0, 1]) ==> "acbd"
  tf.reduce_join(a, [1, 0]) ==> "abcd"
  tf.reduce_join(a, []) ==> [["a", "b"], ["c", "d"]]
  tf.reduce_join(a) = tf.reduce_join(a, [1, 0]) ==> "abcd"
  ```

  Args:
    inputs: A `Tensor` of type `string`.
      The input to be joined.  All reduced indices must have non-zero size.
    reduction_indices: A `Tensor` of type `int32`.
      The dimensions to reduce over.  Dimensions are reduced in the
      order specified.  Omitting `reduction_indices` is equivalent to passing
      `[n-1, n-2, ..., 0]`.  Negative indices from `-n` to `-1` are supported.
    keep_dims: An optional `bool`. Defaults to `False`.
      If `True`, retain reduced dimensions with length `1`.
    separator: An optional `string`. Defaults to `""`.
      The separator to use when joining.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `string`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "ReduceJoin", name, inputs, reduction_indices, "keep_dims",
        keep_dims, "separator", separator)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return reduce_join_eager_fallback(
          inputs, reduction_indices, keep_dims=keep_dims, separator=separator,
          name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if keep_dims is None:
    keep_dims = False
  keep_dims = _execute.make_bool(keep_dims, "keep_dims")
  if separator is None:
    separator = ""
  separator = _execute.make_str(separator, "separator")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "ReduceJoin", inputs=inputs, reduction_indices=reduction_indices,
                      keep_dims=keep_dims, separator=separator, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("keep_dims", _op._get_attr_bool("keep_dims"), "separator",
              _op.get_attr("separator"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "ReduceJoin", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

ReduceJoin = tf_export("raw_ops.ReduceJoin")(_ops.to_raw_op(reduce_join))


def reduce_join_eager_fallback(inputs: Annotated[Any, _atypes.String], reduction_indices: Annotated[Any, _atypes.Int32], keep_dims: bool, separator: str, name, ctx) -> Annotated[Any, _atypes.String]:
  if keep_dims is None:
    keep_dims = False
  keep_dims = _execute.make_bool(keep_dims, "keep_dims")
  if separator is None:
    separator = ""
  separator = _execute.make_str(separator, "separator")
  inputs = _ops.convert_to_tensor(inputs, _dtypes.string)
  reduction_indices = _ops.convert_to_tensor(reduction_indices, _dtypes.int32)
  _inputs_flat = [inputs, reduction_indices]
  _attrs = ("keep_dims", keep_dims, "separator", separator)
  _result = _execute.execute(b"ReduceJoin", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "ReduceJoin", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


def regex_full_match(input: Annotated[Any, _atypes.String], pattern: Annotated[Any, _atypes.String], name=None) -> Annotated[Any, _atypes.Bool]:
  r"""Check if the input matches the regex pattern.

  The input is a string tensor of any shape. The pattern is a scalar
  string tensor which is applied to every element of the input tensor.
  The boolean values (True or False) of the output tensor indicate
  if the input matches the regex pattern provided.

  The pattern follows the re2 syntax (https://github.com/google/re2/wiki/Syntax)

  Examples:

  >>> tf.strings.regex_full_match(["TF lib", "lib TF"], ".*lib$")
  <tf.Tensor: shape=(2,), dtype=bool, numpy=array([ True, False])>
  >>> tf.strings.regex_full_match(["TF lib", "lib TF"], ".*TF$")
  <tf.Tensor: shape=(2,), dtype=bool, numpy=array([False,  True])>

  Args:
    input: A `Tensor` of type `string`.
      A string tensor of the text to be processed.
    pattern: A `Tensor` of type `string`.
      A scalar string tensor containing the regular expression to match the input.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `bool`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "RegexFullMatch", name, input, pattern)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return regex_full_match_eager_fallback(
          input, pattern, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "RegexFullMatch", input=input, pattern=pattern, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ()
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "RegexFullMatch", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

RegexFullMatch = tf_export("raw_ops.RegexFullMatch")(_ops.to_raw_op(regex_full_match))


def regex_full_match_eager_fallback(input: Annotated[Any, _atypes.String], pattern: Annotated[Any, _atypes.String], name, ctx) -> Annotated[Any, _atypes.Bool]:
  input = _ops.convert_to_tensor(input, _dtypes.string)
  pattern = _ops.convert_to_tensor(pattern, _dtypes.string)
  _inputs_flat = [input, pattern]
  _attrs = None
  _result = _execute.execute(b"RegexFullMatch", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "RegexFullMatch", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


def regex_replace(input: Annotated[Any, _atypes.String], pattern: Annotated[Any, _atypes.String], rewrite: Annotated[Any, _atypes.String], replace_global:bool=True, name=None) -> Annotated[Any, _atypes.String]:
  r"""Replaces matches of the `pattern` regular expression in `input` with the
replacement string provided in `rewrite`.

  It follows the re2 syntax (https://github.com/google/re2/wiki/Syntax)

  Args:
    input: A `Tensor` of type `string`. The text to be processed.
    pattern: A `Tensor` of type `string`.
      The regular expression to be matched in the `input` strings.
    rewrite: A `Tensor` of type `string`.
      The rewrite string to be substituted for the `pattern` expression where it is
      matched in the `input` strings.
    replace_global: An optional `bool`. Defaults to `True`.
      If True, the replacement is global (that is, all matches of the `pattern` regular
      expression in each input string are rewritten), otherwise the `rewrite`
      substitution is only made for the first `pattern` match.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `string`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "RegexReplace", name, input, pattern, rewrite, "replace_global",
        replace_global)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return regex_replace_eager_fallback(
          input, pattern, rewrite, replace_global=replace_global, name=name,
          ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if replace_global is None:
    replace_global = True
  replace_global = _execute.make_bool(replace_global, "replace_global")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "RegexReplace", input=input, pattern=pattern, rewrite=rewrite,
                        replace_global=replace_global, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("replace_global", _op._get_attr_bool("replace_global"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "RegexReplace", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

RegexReplace = tf_export("raw_ops.RegexReplace")(_ops.to_raw_op(regex_replace))


def regex_replace_eager_fallback(input: Annotated[Any, _atypes.String], pattern: Annotated[Any, _atypes.String], rewrite: Annotated[Any, _atypes.String], replace_global: bool, name, ctx) -> Annotated[Any, _atypes.String]:
  if replace_global is None:
    replace_global = True
  replace_global = _execute.make_bool(replace_global, "replace_global")
  input = _ops.convert_to_tensor(input, _dtypes.string)
  pattern = _ops.convert_to_tensor(pattern, _dtypes.string)
  rewrite = _ops.convert_to_tensor(rewrite, _dtypes.string)
  _inputs_flat = [input, pattern, rewrite]
  _attrs = ("replace_global", replace_global)
  _result = _execute.execute(b"RegexReplace", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "RegexReplace", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


def static_regex_full_match(input: Annotated[Any, _atypes.String], pattern: str, name=None) -> Annotated[Any, _atypes.Bool]:
  r"""Check if the input matches the regex pattern.

  The input is a string tensor of any shape. The pattern is the
  regular expression to be matched with every element of the input tensor.
  The boolean values (True or False) of the output tensor indicate
  if the input matches the regex pattern provided.

  The pattern follows the re2 syntax (https://github.com/google/re2/wiki/Syntax)

  Args:
    input: A `Tensor` of type `string`.
      A string tensor of the text to be processed.
    pattern: A `string`. The regular expression to match the input.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `bool`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "StaticRegexFullMatch", name, input, "pattern", pattern)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return static_regex_full_match_eager_fallback(
          input, pattern=pattern, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  pattern = _execute.make_str(pattern, "pattern")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "StaticRegexFullMatch", input=input, pattern=pattern, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("pattern", _op.get_attr("pattern"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "StaticRegexFullMatch", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

StaticRegexFullMatch = tf_export("raw_ops.StaticRegexFullMatch")(_ops.to_raw_op(static_regex_full_match))


def static_regex_full_match_eager_fallback(input: Annotated[Any, _atypes.String], pattern: str, name, ctx) -> Annotated[Any, _atypes.Bool]:
  pattern = _execute.make_str(pattern, "pattern")
  input = _ops.convert_to_tensor(input, _dtypes.string)
  _inputs_flat = [input]
  _attrs = ("pattern", pattern)
  _result = _execute.execute(b"StaticRegexFullMatch", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "StaticRegexFullMatch", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


def static_regex_replace(input: Annotated[Any, _atypes.String], pattern: str, rewrite: str, replace_global:bool=True, name=None) -> Annotated[Any, _atypes.String]:
  r"""Replaces the match of pattern in input with rewrite.

  It follows the re2 syntax (https://github.com/google/re2/wiki/Syntax)

  Args:
    input: A `Tensor` of type `string`. The text to be processed.
    pattern: A `string`. The regular expression to match the input.
    rewrite: A `string`. The rewrite to be applied to the matched expression.
    replace_global: An optional `bool`. Defaults to `True`.
      If True, the replacement is global, otherwise the replacement
      is done only on the first match.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `string`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "StaticRegexReplace", name, input, "pattern", pattern,
        "rewrite", rewrite, "replace_global", replace_global)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return static_regex_replace_eager_fallback(
          input, pattern=pattern, rewrite=rewrite,
          replace_global=replace_global, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  pattern = _execute.make_str(pattern, "pattern")
  rewrite = _execute.make_str(rewrite, "rewrite")
  if replace_global is None:
    replace_global = True
  replace_global = _execute.make_bool(replace_global, "replace_global")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "StaticRegexReplace", input=input, pattern=pattern, rewrite=rewrite,
                              replace_global=replace_global, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("pattern", _op.get_attr("pattern"), "rewrite",
              _op.get_attr("rewrite"), "replace_global",
              _op._get_attr_bool("replace_global"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "StaticRegexReplace", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

StaticRegexReplace = tf_export("raw_ops.StaticRegexReplace")(_ops.to_raw_op(static_regex_replace))


def static_regex_replace_eager_fallback(input: Annotated[Any, _atypes.String], pattern: str, rewrite: str, replace_global: bool, name, ctx) -> Annotated[Any, _atypes.String]:
  pattern = _execute.make_str(pattern, "pattern")
  rewrite = _execute.make_str(rewrite, "rewrite")
  if replace_global is None:
    replace_global = True
  replace_global = _execute.make_bool(replace_global, "replace_global")
  input = _ops.convert_to_tensor(input, _dtypes.string)
  _inputs_flat = [input]
  _attrs = ("pattern", pattern, "rewrite", rewrite, "replace_global",
  replace_global)
  _result = _execute.execute(b"StaticRegexReplace", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "StaticRegexReplace", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


def string_format(inputs, template:str="%s", placeholder:str="%s", summarize:int=3, name=None) -> Annotated[Any, _atypes.String]:
  r"""Formats a string template using a list of tensors.

  Formats a string template using a list of tensors, pretty-printing tensor summaries.

  Args:
    inputs: A list of `Tensor` objects.
      The list of tensors to format into the placeholder string.
    template: An optional `string`. Defaults to `"%s"`.
      A string, the template to format tensor summaries into.
    placeholder: An optional `string`. Defaults to `"%s"`.
      A string, at each placeholder in the template a subsequent tensor summary will be inserted.
    summarize: An optional `int`. Defaults to `3`.
      When formatting the tensor summaries print the first and last summarize entries of each tensor dimension.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `string`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "StringFormat", name, inputs, "template", template,
        "placeholder", placeholder, "summarize", summarize)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return string_format_eager_fallback(
          inputs, template=template, placeholder=placeholder,
          summarize=summarize, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if template is None:
    template = "%s"
  template = _execute.make_str(template, "template")
  if placeholder is None:
    placeholder = "%s"
  placeholder = _execute.make_str(placeholder, "placeholder")
  if summarize is None:
    summarize = 3
  summarize = _execute.make_int(summarize, "summarize")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "StringFormat", inputs=inputs, template=template,
                        placeholder=placeholder, summarize=summarize,
                        name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op.get_attr("T"), "template", _op.get_attr("template"),
              "placeholder", _op.get_attr("placeholder"), "summarize",
              _op._get_attr_int("summarize"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "StringFormat", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

StringFormat = tf_export("raw_ops.StringFormat")(_ops.to_raw_op(string_format))


def string_format_eager_fallback(inputs, template: str, placeholder: str, summarize: int, name, ctx) -> Annotated[Any, _atypes.String]:
  if template is None:
    template = "%s"
  template = _execute.make_str(template, "template")
  if placeholder is None:
    placeholder = "%s"
  placeholder = _execute.make_str(placeholder, "placeholder")
  if summarize is None:
    summarize = 3
  summarize = _execute.make_int(summarize, "summarize")
  _attr_T, inputs = _execute.convert_to_mixed_eager_tensors(inputs, ctx)
  _inputs_flat = list(inputs)
  _attrs = ("T", _attr_T, "template", template, "placeholder", placeholder,
  "summarize", summarize)
  _result = _execute.execute(b"StringFormat", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "StringFormat", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


def string_join(inputs: Annotated[List[Any], _atypes.String], separator:str="", name=None) -> Annotated[Any, _atypes.String]:
  r"""Joins the strings in the given list of string tensors into one tensor;

  with the given separator (default is an empty separator).

  Examples:

  >>> s = ["hello", "world", "tensorflow"]
  >>> tf.strings.join(s, " ")
  <tf.Tensor: shape=(), dtype=string, numpy=b'hello world tensorflow'>

  Args:
    inputs: A list of `Tensor` objects with type `string`.
      A list of string tensors.  The tensors must all have the same shape,
      or be scalars.  Scalars may be mixed in; these will be broadcast to the shape
      of non-scalar inputs.
    separator: An optional `string`. Defaults to `""`.
      string, an optional join separator.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `string`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "StringJoin", name, inputs, "separator", separator)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return string_join_eager_fallback(
          inputs, separator=separator, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if not isinstance(inputs, (list, tuple)):
    raise TypeError(
        "Expected list for 'inputs' argument to "
        "'string_join' Op, not %r." % inputs)
  _attr_N = len(inputs)
  if separator is None:
    separator = ""
  separator = _execute.make_str(separator, "separator")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "StringJoin", inputs=inputs, separator=separator, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("N", _op._get_attr_int("N"), "separator",
              _op.get_attr("separator"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "StringJoin", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

StringJoin = tf_export("raw_ops.StringJoin")(_ops.to_raw_op(string_join))


def string_join_eager_fallback(inputs: Annotated[List[Any], _atypes.String], separator: str, name, ctx) -> Annotated[Any, _atypes.String]:
  if not isinstance(inputs, (list, tuple)):
    raise TypeError(
        "Expected list for 'inputs' argument to "
        "'string_join' Op, not %r." % inputs)
  _attr_N = len(inputs)
  if separator is None:
    separator = ""
  separator = _execute.make_str(separator, "separator")
  inputs = _ops.convert_n_to_tensor(inputs, _dtypes.string)
  _inputs_flat = list(inputs)
  _attrs = ("N", _attr_N, "separator", separator)
  _result = _execute.execute(b"StringJoin", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "StringJoin", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


def string_length(input: Annotated[Any, _atypes.String], unit:str="BYTE", name=None) -> Annotated[Any, _atypes.Int32]:
  r"""String lengths of `input`.

  Computes the length of each string given in the input tensor.

  >>> strings = tf.constant(['Hello','TensorFlow', '\U0001F642'])
  >>> tf.strings.length(strings).numpy() # default counts bytes
  array([ 5, 10, 4], dtype=int32)
  >>> tf.strings.length(strings, unit="UTF8_CHAR").numpy()
  array([ 5, 10, 1], dtype=int32)

  Args:
    input: A `Tensor` of type `string`.
      The strings for which to compute the length for each element.
    unit: An optional `string` from: `"BYTE", "UTF8_CHAR"`. Defaults to `"BYTE"`.
      The unit that is counted to compute string length.  One of: `"BYTE"` (for
      the number of bytes in each string) or `"UTF8_CHAR"` (for the number of UTF-8
      encoded Unicode code points in each string).  Results are undefined
      if `unit=UTF8_CHAR` and the `input` strings do not contain structurally
      valid UTF-8.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `int32`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "StringLength", name, input, "unit", unit)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return string_length_eager_fallback(
          input, unit=unit, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if unit is None:
    unit = "BYTE"
  unit = _execute.make_str(unit, "unit")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "StringLength", input=input, unit=unit, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("unit", _op.get_attr("unit"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "StringLength", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

StringLength = tf_export("raw_ops.StringLength")(_ops.to_raw_op(string_length))


def string_length_eager_fallback(input: Annotated[Any, _atypes.String], unit: str, name, ctx) -> Annotated[Any, _atypes.Int32]:
  if unit is None:
    unit = "BYTE"
  unit = _execute.make_str(unit, "unit")
  input = _ops.convert_to_tensor(input, _dtypes.string)
  _inputs_flat = [input]
  _attrs = ("unit", unit)
  _result = _execute.execute(b"StringLength", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "StringLength", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('strings.lower')
def string_lower(input: Annotated[Any, _atypes.String], encoding:str="", name=None) -> Annotated[Any, _atypes.String]:
  r"""Converts all uppercase characters into their respective lowercase replacements.

  Example:

  >>> tf.strings.lower("CamelCase string and ALL CAPS")
  <tf.Tensor: shape=(), dtype=string, numpy=b'camelcase string and all caps'>

  Args:
    input: A `Tensor` of type `string`. The input to be lower-cased.
    encoding: An optional `string`. Defaults to `""`.
      Character encoding of `input`. Allowed values are '' and 'utf-8'.
      Value '' is interpreted as ASCII.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `string`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "StringLower", name, input, "encoding", encoding)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_string_lower(
          (input, encoding, name,), None)
      if _result is not NotImplemented:
        return _result
      return string_lower_eager_fallback(
          input, encoding=encoding, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            string_lower, (), dict(input=input, encoding=encoding, name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_string_lower(
        (input, encoding, name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  if encoding is None:
    encoding = ""
  encoding = _execute.make_str(encoding, "encoding")
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "StringLower", input=input, encoding=encoding, name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          string_lower, (), dict(input=input, encoding=encoding, name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("encoding", _op.get_attr("encoding"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "StringLower", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

StringLower = tf_export("raw_ops.StringLower")(_ops.to_raw_op(string_lower))
_dispatcher_for_string_lower = string_lower._tf_type_based_dispatcher.Dispatch


def string_lower_eager_fallback(input: Annotated[Any, _atypes.String], encoding: str, name, ctx) -> Annotated[Any, _atypes.String]:
  if encoding is None:
    encoding = ""
  encoding = _execute.make_str(encoding, "encoding")
  input = _ops.convert_to_tensor(input, _dtypes.string)
  _inputs_flat = [input]
  _attrs = ("encoding", encoding)
  _result = _execute.execute(b"StringLower", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "StringLower", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

_StringNGramsOutput = collections.namedtuple(
    "StringNGrams",
    ["ngrams", "ngrams_splits"])


TV_StringNGrams_Tsplits = TypeVar("TV_StringNGrams_Tsplits", _atypes.Int32, _atypes.Int64)

def string_n_grams(data: Annotated[Any, _atypes.String], data_splits: Annotated[Any, TV_StringNGrams_Tsplits], separator: str, ngram_widths, left_pad: str, right_pad: str, pad_width: int, preserve_short_sequences: bool, name=None):
  r"""Creates ngrams from ragged string data.

  This op accepts a ragged tensor with 1 ragged dimension containing only
  strings and outputs a ragged tensor with 1 ragged dimension containing ngrams
  of that string, joined along the innermost axis.

  Args:
    data: A `Tensor` of type `string`.
      The values tensor of the ragged string tensor to make ngrams out of. Must be a
      1D string tensor.
    data_splits: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      The splits tensor of the ragged string tensor to make ngrams out of.
    separator: A `string`.
      The string to append between elements of the token. Use "" for no separator.
    ngram_widths: A list of `ints`. The sizes of the ngrams to create.
    left_pad: A `string`.
      The string to use to pad the left side of the ngram sequence. Only used if
      pad_width != 0.
    right_pad: A `string`.
      The string to use to pad the right side of the ngram sequence. Only used if
      pad_width != 0.
    pad_width: An `int`.
      The number of padding elements to add to each side of each
      sequence. Note that padding will never be greater than 'ngram_widths'-1
      regardless of this value. If `pad_width=-1`, then add `max(ngram_widths)-1`
      elements.
    preserve_short_sequences: A `bool`.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (ngrams, ngrams_splits).

    ngrams: A `Tensor` of type `string`.
    ngrams_splits: A `Tensor`. Has the same type as `data_splits`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "StringNGrams", name, data, data_splits, "separator", separator,
        "ngram_widths", ngram_widths, "left_pad", left_pad, "right_pad",
        right_pad, "pad_width", pad_width, "preserve_short_sequences",
        preserve_short_sequences)
      _result = _StringNGramsOutput._make(_result)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return string_n_grams_eager_fallback(
          data, data_splits, separator=separator, ngram_widths=ngram_widths,
          left_pad=left_pad, right_pad=right_pad, pad_width=pad_width,
          preserve_short_sequences=preserve_short_sequences, name=name,
          ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  separator = _execute.make_str(separator, "separator")
  if not isinstance(ngram_widths, (list, tuple)):
    raise TypeError(
        "Expected list for 'ngram_widths' argument to "
        "'string_n_grams' Op, not %r." % ngram_widths)
  ngram_widths = [_execute.make_int(_i, "ngram_widths") for _i in ngram_widths]
  left_pad = _execute.make_str(left_pad, "left_pad")
  right_pad = _execute.make_str(right_pad, "right_pad")
  pad_width = _execute.make_int(pad_width, "pad_width")
  preserve_short_sequences = _execute.make_bool(preserve_short_sequences, "preserve_short_sequences")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "StringNGrams", data=data, data_splits=data_splits,
                        separator=separator, ngram_widths=ngram_widths,
                        left_pad=left_pad, right_pad=right_pad,
                        pad_width=pad_width,
                        preserve_short_sequences=preserve_short_sequences,
                        name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("separator", _op.get_attr("separator"), "ngram_widths",
              _op.get_attr("ngram_widths"), "left_pad",
              _op.get_attr("left_pad"), "right_pad",
              _op.get_attr("right_pad"), "pad_width",
              _op._get_attr_int("pad_width"), "preserve_short_sequences",
              _op._get_attr_bool("preserve_short_sequences"), "Tsplits",
              _op._get_attr_type("Tsplits"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "StringNGrams", _inputs_flat, _attrs, _result)
  _result = _StringNGramsOutput._make(_result)
  return _result

StringNGrams = tf_export("raw_ops.StringNGrams")(_ops.to_raw_op(string_n_grams))


def string_n_grams_eager_fallback(data: Annotated[Any, _atypes.String], data_splits: Annotated[Any, TV_StringNGrams_Tsplits], separator: str, ngram_widths, left_pad: str, right_pad: str, pad_width: int, preserve_short_sequences: bool, name, ctx):
  separator = _execute.make_str(separator, "separator")
  if not isinstance(ngram_widths, (list, tuple)):
    raise TypeError(
        "Expected list for 'ngram_widths' argument to "
        "'string_n_grams' Op, not %r." % ngram_widths)
  ngram_widths = [_execute.make_int(_i, "ngram_widths") for _i in ngram_widths]
  left_pad = _execute.make_str(left_pad, "left_pad")
  right_pad = _execute.make_str(right_pad, "right_pad")
  pad_width = _execute.make_int(pad_width, "pad_width")
  preserve_short_sequences = _execute.make_bool(preserve_short_sequences, "preserve_short_sequences")
  _attr_Tsplits, (data_splits,) = _execute.args_to_matching_eager([data_splits], ctx, [_dtypes.int32, _dtypes.int64, ], _dtypes.int64)
  data = _ops.convert_to_tensor(data, _dtypes.string)
  _inputs_flat = [data, data_splits]
  _attrs = ("separator", separator, "ngram_widths", ngram_widths, "left_pad",
  left_pad, "right_pad", right_pad, "pad_width", pad_width,
  "preserve_short_sequences", preserve_short_sequences, "Tsplits",
  _attr_Tsplits)
  _result = _execute.execute(b"StringNGrams", 2, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "StringNGrams", _inputs_flat, _attrs, _result)
  _result = _StringNGramsOutput._make(_result)
  return _result

_StringSplitOutput = collections.namedtuple(
    "StringSplit",
    ["indices", "values", "shape"])


def string_split(input: Annotated[Any, _atypes.String], delimiter: Annotated[Any, _atypes.String], skip_empty:bool=True, name=None):
  r"""Split elements of `input` based on `delimiter` into a `SparseTensor`.

  Let N be the size of source (typically N will be the batch size). Split each
  element of `input` based on `delimiter` and return a `SparseTensor`
  containing the splitted tokens. Empty tokens are ignored.

  `delimiter` can be empty, or a string of split characters. If `delimiter` is an
   empty string, each element of `input` is split into individual single-byte
   character strings, including splitting of UTF-8 multibyte sequences. Otherwise
   every character of `delimiter` is a potential split point.

  For example:
    N = 2, input[0] is 'hello world' and input[1] is 'a b c', then the output
    will be

    indices = [0, 0;
               0, 1;
               1, 0;
               1, 1;
               1, 2]
    shape = [2, 3]
    values = ['hello', 'world', 'a', 'b', 'c']

  Args:
    input: A `Tensor` of type `string`. 1-D. Strings to split.
    delimiter: A `Tensor` of type `string`.
      0-D. Delimiter characters (bytes), or empty string.
    skip_empty: An optional `bool`. Defaults to `True`.
      A `bool`. If `True`, skip the empty strings from the result.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (indices, values, shape).

    indices: A `Tensor` of type `int64`.
    values: A `Tensor` of type `string`.
    shape: A `Tensor` of type `int64`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "StringSplit", name, input, delimiter, "skip_empty", skip_empty)
      _result = _StringSplitOutput._make(_result)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return string_split_eager_fallback(
          input, delimiter, skip_empty=skip_empty, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if skip_empty is None:
    skip_empty = True
  skip_empty = _execute.make_bool(skip_empty, "skip_empty")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "StringSplit", input=input, delimiter=delimiter,
                       skip_empty=skip_empty, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("skip_empty", _op._get_attr_bool("skip_empty"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "StringSplit", _inputs_flat, _attrs, _result)
  _result = _StringSplitOutput._make(_result)
  return _result

StringSplit = tf_export("raw_ops.StringSplit")(_ops.to_raw_op(string_split))


def string_split_eager_fallback(input: Annotated[Any, _atypes.String], delimiter: Annotated[Any, _atypes.String], skip_empty: bool, name, ctx):
  if skip_empty is None:
    skip_empty = True
  skip_empty = _execute.make_bool(skip_empty, "skip_empty")
  input = _ops.convert_to_tensor(input, _dtypes.string)
  delimiter = _ops.convert_to_tensor(delimiter, _dtypes.string)
  _inputs_flat = [input, delimiter]
  _attrs = ("skip_empty", skip_empty)
  _result = _execute.execute(b"StringSplit", 3, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "StringSplit", _inputs_flat, _attrs, _result)
  _result = _StringSplitOutput._make(_result)
  return _result

_StringSplitV2Output = collections.namedtuple(
    "StringSplitV2",
    ["indices", "values", "shape"])


def string_split_v2(input: Annotated[Any, _atypes.String], sep: Annotated[Any, _atypes.String], maxsplit:int=-1, name=None):
  r"""Split elements of `source` based on `sep` into a `SparseTensor`.

  Let N be the size of source (typically N will be the batch size). Split each
  element of `source` based on `sep` and return a `SparseTensor`
  containing the split tokens. Empty tokens are ignored.

  For example, N = 2, source[0] is 'hello world' and source[1] is 'a b c',
  then the output will be
  ```
  st.indices = [0, 0;
                0, 1;
                1, 0;
                1, 1;
                1, 2]
  st.shape = [2, 3]
  st.values = ['hello', 'world', 'a', 'b', 'c']
  ```

  If `sep` is given, consecutive delimiters are not grouped together and are
  deemed to delimit empty strings. For example, source of `"1<>2<><>3"` and
  sep of `"<>"` returns `["1", "2", "", "3"]`. If `sep` is None or an empty
  string, consecutive whitespace are regarded as a single separator, and the
  result will contain no empty strings at the startor end if the string has
  leading or trailing whitespace.

  Note that the above mentioned behavior matches python's str.split.

  Args:
    input: A `Tensor` of type `string`.
      `1-D` string `Tensor`, the strings to split.
    sep: A `Tensor` of type `string`.
      `0-D` string `Tensor`, the delimiter character.
    maxsplit: An optional `int`. Defaults to `-1`.
      An `int`. If `maxsplit > 0`, limit of the split of the result.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (indices, values, shape).

    indices: A `Tensor` of type `int64`.
    values: A `Tensor` of type `string`.
    shape: A `Tensor` of type `int64`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "StringSplitV2", name, input, sep, "maxsplit", maxsplit)
      _result = _StringSplitV2Output._make(_result)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return string_split_v2_eager_fallback(
          input, sep, maxsplit=maxsplit, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if maxsplit is None:
    maxsplit = -1
  maxsplit = _execute.make_int(maxsplit, "maxsplit")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "StringSplitV2", input=input, sep=sep, maxsplit=maxsplit, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("maxsplit", _op._get_attr_int("maxsplit"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "StringSplitV2", _inputs_flat, _attrs, _result)
  _result = _StringSplitV2Output._make(_result)
  return _result

StringSplitV2 = tf_export("raw_ops.StringSplitV2")(_ops.to_raw_op(string_split_v2))


def string_split_v2_eager_fallback(input: Annotated[Any, _atypes.String], sep: Annotated[Any, _atypes.String], maxsplit: int, name, ctx):
  if maxsplit is None:
    maxsplit = -1
  maxsplit = _execute.make_int(maxsplit, "maxsplit")
  input = _ops.convert_to_tensor(input, _dtypes.string)
  sep = _ops.convert_to_tensor(sep, _dtypes.string)
  _inputs_flat = [input, sep]
  _attrs = ("maxsplit", maxsplit)
  _result = _execute.execute(b"StringSplitV2", 3, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "StringSplitV2", _inputs_flat, _attrs, _result)
  _result = _StringSplitV2Output._make(_result)
  return _result


@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('strings.strip', v1=['strings.strip', 'string_strip'])
@deprecated_endpoints('string_strip')
def string_strip(input: Annotated[Any, _atypes.String], name=None) -> Annotated[Any, _atypes.String]:
  r"""Strip leading and trailing whitespaces from the Tensor.

  Examples:

  >>> tf.strings.strip(["\nTensorFlow", "     The python library    "]).numpy()
  array([b'TensorFlow', b'The python library'], dtype=object)

  Args:
    input: A `Tensor` of type `string`. A string `Tensor` of any shape.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `string`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "StringStrip", name, input)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_string_strip(
          (input, name,), None)
      if _result is not NotImplemented:
        return _result
      return string_strip_eager_fallback(
          input, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            string_strip, (), dict(input=input, name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_string_strip(
        (input, name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "StringStrip", input=input, name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          string_strip, (), dict(input=input, name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ()
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "StringStrip", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

StringStrip = tf_export("raw_ops.StringStrip")(_ops.to_raw_op(string_strip))
_dispatcher_for_string_strip = string_strip._tf_type_based_dispatcher.Dispatch


def string_strip_eager_fallback(input: Annotated[Any, _atypes.String], name, ctx) -> Annotated[Any, _atypes.String]:
  input = _ops.convert_to_tensor(input, _dtypes.string)
  _inputs_flat = [input]
  _attrs = None
  _result = _execute.execute(b"StringStrip", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "StringStrip", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


def string_to_hash_bucket(string_tensor: Annotated[Any, _atypes.String], num_buckets: int, name=None) -> Annotated[Any, _atypes.Int64]:
  r"""Converts each string in the input Tensor to its hash mod by a number of buckets.

  The hash function is deterministic on the content of the string within the
  process.

  Note that the hash function may change from time to time.
  This functionality will be deprecated and it's recommended to use
  `tf.string_to_hash_bucket_fast()` or `tf.string_to_hash_bucket_strong()`.

  Args:
    string_tensor: A `Tensor` of type `string`.
    num_buckets: An `int` that is `>= 1`. The number of buckets.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `int64`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "StringToHashBucket", name, string_tensor, "num_buckets",
        num_buckets)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return string_to_hash_bucket_eager_fallback(
          string_tensor, num_buckets=num_buckets, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  num_buckets = _execute.make_int(num_buckets, "num_buckets")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "StringToHashBucket", string_tensor=string_tensor,
                              num_buckets=num_buckets, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("num_buckets", _op._get_attr_int("num_buckets"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "StringToHashBucket", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

StringToHashBucket = tf_export("raw_ops.StringToHashBucket")(_ops.to_raw_op(string_to_hash_bucket))


def string_to_hash_bucket_eager_fallback(string_tensor: Annotated[Any, _atypes.String], num_buckets: int, name, ctx) -> Annotated[Any, _atypes.Int64]:
  num_buckets = _execute.make_int(num_buckets, "num_buckets")
  string_tensor = _ops.convert_to_tensor(string_tensor, _dtypes.string)
  _inputs_flat = [string_tensor]
  _attrs = ("num_buckets", num_buckets)
  _result = _execute.execute(b"StringToHashBucket", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "StringToHashBucket", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('strings.to_hash_bucket_fast', v1=['strings.to_hash_bucket_fast', 'string_to_hash_bucket_fast'])
@deprecated_endpoints('string_to_hash_bucket_fast')
def string_to_hash_bucket_fast(input: Annotated[Any, _atypes.String], num_buckets: int, name=None) -> Annotated[Any, _atypes.Int64]:
  r"""Converts each string in the input Tensor to its hash mod by a number of buckets.

  The hash function is deterministic on the content of the string within the
  process and will never change. However, it is not suitable for cryptography.
  This function may be used when CPU time is scarce and inputs are trusted or
  unimportant. There is a risk of adversaries constructing inputs that all hash
  to the same bucket. To prevent this problem, use a strong hash function with
  `tf.string_to_hash_bucket_strong`.

  Examples:

  >>> tf.strings.to_hash_bucket_fast(["Hello", "TensorFlow", "2.x"], 3).numpy()
  array([0, 2, 2])

  Args:
    input: A `Tensor` of type `string`. The strings to assign a hash bucket.
    num_buckets: An `int` that is `>= 1`. The number of buckets.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `int64`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "StringToHashBucketFast", name, input, "num_buckets",
        num_buckets)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_string_to_hash_bucket_fast(
          (input, num_buckets, name,), None)
      if _result is not NotImplemented:
        return _result
      return string_to_hash_bucket_fast_eager_fallback(
          input, num_buckets=num_buckets, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            string_to_hash_bucket_fast, (), dict(input=input,
                                                 num_buckets=num_buckets,
                                                 name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_string_to_hash_bucket_fast(
        (input, num_buckets, name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  num_buckets = _execute.make_int(num_buckets, "num_buckets")
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "StringToHashBucketFast", input=input, num_buckets=num_buckets,
                                  name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          string_to_hash_bucket_fast, (), dict(input=input,
                                               num_buckets=num_buckets,
                                               name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("num_buckets", _op._get_attr_int("num_buckets"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "StringToHashBucketFast", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

StringToHashBucketFast = tf_export("raw_ops.StringToHashBucketFast")(_ops.to_raw_op(string_to_hash_bucket_fast))
_dispatcher_for_string_to_hash_bucket_fast = string_to_hash_bucket_fast._tf_type_based_dispatcher.Dispatch


def string_to_hash_bucket_fast_eager_fallback(input: Annotated[Any, _atypes.String], num_buckets: int, name, ctx) -> Annotated[Any, _atypes.Int64]:
  num_buckets = _execute.make_int(num_buckets, "num_buckets")
  input = _ops.convert_to_tensor(input, _dtypes.string)
  _inputs_flat = [input]
  _attrs = ("num_buckets", num_buckets)
  _result = _execute.execute(b"StringToHashBucketFast", 1,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "StringToHashBucketFast", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('strings.to_hash_bucket_strong', v1=['strings.to_hash_bucket_strong', 'string_to_hash_bucket_strong'])
@deprecated_endpoints('string_to_hash_bucket_strong')
def string_to_hash_bucket_strong(input: Annotated[Any, _atypes.String], num_buckets: int, key, name=None) -> Annotated[Any, _atypes.Int64]:
  r"""Converts each string in the input Tensor to its hash mod by a number of buckets.

  The hash function is deterministic on the content of the string within the
  process. The hash function is a keyed hash function, where attribute `key`
  defines the key of the hash function. `key` is an array of 2 elements.

  A strong hash is important when inputs may be malicious, e.g. URLs with
  additional components. Adversaries could try to make their inputs hash to the
  same bucket for a denial-of-service attack or to skew the results. A strong
  hash can be used to make it difficult to find inputs with a skewed hash value
  distribution over buckets. This requires that the hash function is
  seeded by a high-entropy (random) "key" unknown to the adversary.

  The additional robustness comes at a cost of roughly 4x higher compute
  time than `tf.string_to_hash_bucket_fast`.

  Examples:

  >>> tf.strings.to_hash_bucket_strong(["Hello", "TF"], 3, [1, 2]).numpy()
  array([2, 0])

  Args:
    input: A `Tensor` of type `string`. The strings to assign a hash bucket.
    num_buckets: An `int` that is `>= 1`. The number of buckets.
    key: A list of `ints`.
      The key used to seed the hash function, passed as a list of two uint64
      elements.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `int64`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "StringToHashBucketStrong", name, input, "num_buckets",
        num_buckets, "key", key)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_string_to_hash_bucket_strong(
          (input, num_buckets, key, name,), None)
      if _result is not NotImplemented:
        return _result
      return string_to_hash_bucket_strong_eager_fallback(
          input, num_buckets=num_buckets, key=key, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            string_to_hash_bucket_strong, (), dict(input=input,
                                                   num_buckets=num_buckets,
                                                   key=key, name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_string_to_hash_bucket_strong(
        (input, num_buckets, key, name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  num_buckets = _execute.make_int(num_buckets, "num_buckets")
  if not isinstance(key, (list, tuple)):
    raise TypeError(
        "Expected list for 'key' argument to "
        "'string_to_hash_bucket_strong' Op, not %r." % key)
  key = [_execute.make_int(_i, "key") for _i in key]
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "StringToHashBucketStrong", input=input, num_buckets=num_buckets,
                                    key=key, name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          string_to_hash_bucket_strong, (), dict(input=input,
                                                 num_buckets=num_buckets,
                                                 key=key, name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("num_buckets", _op._get_attr_int("num_buckets"), "key",
              _op.get_attr("key"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "StringToHashBucketStrong", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

StringToHashBucketStrong = tf_export("raw_ops.StringToHashBucketStrong")(_ops.to_raw_op(string_to_hash_bucket_strong))
_dispatcher_for_string_to_hash_bucket_strong = string_to_hash_bucket_strong._tf_type_based_dispatcher.Dispatch


def string_to_hash_bucket_strong_eager_fallback(input: Annotated[Any, _atypes.String], num_buckets: int, key, name, ctx) -> Annotated[Any, _atypes.Int64]:
  num_buckets = _execute.make_int(num_buckets, "num_buckets")
  if not isinstance(key, (list, tuple)):
    raise TypeError(
        "Expected list for 'key' argument to "
        "'string_to_hash_bucket_strong' Op, not %r." % key)
  key = [_execute.make_int(_i, "key") for _i in key]
  input = _ops.convert_to_tensor(input, _dtypes.string)
  _inputs_flat = [input]
  _attrs = ("num_buckets", num_buckets, "key", key)
  _result = _execute.execute(b"StringToHashBucketStrong", 1,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "StringToHashBucketStrong", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('strings.upper')
def string_upper(input: Annotated[Any, _atypes.String], encoding:str="", name=None) -> Annotated[Any, _atypes.String]:
  r"""Converts all lowercase characters into their respective uppercase replacements.

  Example:

  >>> tf.strings.upper("CamelCase string and ALL CAPS")
  <tf.Tensor: shape=(), dtype=string, numpy=b'CAMELCASE STRING AND ALL CAPS'>

  Args:
    input: A `Tensor` of type `string`. The input to be upper-cased.
    encoding: An optional `string`. Defaults to `""`.
      Character encoding of `input`. Allowed values are '' and 'utf-8'.
      Value '' is interpreted as ASCII.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `string`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "StringUpper", name, input, "encoding", encoding)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_string_upper(
          (input, encoding, name,), None)
      if _result is not NotImplemented:
        return _result
      return string_upper_eager_fallback(
          input, encoding=encoding, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            string_upper, (), dict(input=input, encoding=encoding, name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_string_upper(
        (input, encoding, name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  if encoding is None:
    encoding = ""
  encoding = _execute.make_str(encoding, "encoding")
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "StringUpper", input=input, encoding=encoding, name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          string_upper, (), dict(input=input, encoding=encoding, name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("encoding", _op.get_attr("encoding"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "StringUpper", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

StringUpper = tf_export("raw_ops.StringUpper")(_ops.to_raw_op(string_upper))
_dispatcher_for_string_upper = string_upper._tf_type_based_dispatcher.Dispatch


def string_upper_eager_fallback(input: Annotated[Any, _atypes.String], encoding: str, name, ctx) -> Annotated[Any, _atypes.String]:
  if encoding is None:
    encoding = ""
  encoding = _execute.make_str(encoding, "encoding")
  input = _ops.convert_to_tensor(input, _dtypes.string)
  _inputs_flat = [input]
  _attrs = ("encoding", encoding)
  _result = _execute.execute(b"StringUpper", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "StringUpper", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_Substr_T = TypeVar("TV_Substr_T", _atypes.Int32, _atypes.Int64)

def substr(input: Annotated[Any, _atypes.String], pos: Annotated[Any, TV_Substr_T], len: Annotated[Any, TV_Substr_T], unit:str="BYTE", name=None) -> Annotated[Any, _atypes.String]:
  r"""Return substrings from `Tensor` of strings.

  For each string in the input `Tensor`, creates a substring starting at index
  `pos` with a total length of `len`.

  If `len` defines a substring that would extend beyond the length of the input
  string, or if `len` is negative, then as many characters as possible are used.

  A negative `pos` indicates distance within the string backwards from the end.

  If `pos` specifies an index which is out of range for any of the input strings,
  then an `InvalidArgumentError` is thrown.

  `pos` and `len` must have the same shape, otherwise a `ValueError` is thrown on
  Op creation.

  *NOTE*: `Substr` supports broadcasting up to two dimensions. More about
  broadcasting
  [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)

  ---

  Examples

  Using scalar `pos` and `len`:

  ```python
  input = [b'Hello', b'World']
  position = 1
  length = 3

  output = [b'ell', b'orl']
  ```

  Using `pos` and `len` with same shape as `input`:

  ```python
  input = [[b'ten', b'eleven', b'twelve'],
           [b'thirteen', b'fourteen', b'fifteen'],
           [b'sixteen', b'seventeen', b'eighteen']]
  position = [[1, 2, 3],
              [1, 2, 3],
              [1, 2, 3]]
  length =   [[2, 3, 4],
              [4, 3, 2],
              [5, 5, 5]]

  output = [[b'en', b'eve', b'lve'],
            [b'hirt', b'urt', b'te'],
            [b'ixtee', b'vente', b'hteen']]
  ```

  Broadcasting `pos` and `len` onto `input`:

  ```
  input = [[b'ten', b'eleven', b'twelve'],
           [b'thirteen', b'fourteen', b'fifteen'],
           [b'sixteen', b'seventeen', b'eighteen'],
           [b'nineteen', b'twenty', b'twentyone']]
  position = [1, 2, 3]
  length =   [1, 2, 3]

  output = [[b'e', b'ev', b'lve'],
            [b'h', b'ur', b'tee'],
            [b'i', b've', b'hte'],
            [b'i', b'en', b'nty']]
  ```

  Broadcasting `input` onto `pos` and `len`:

  ```
  input = b'thirteen'
  position = [1, 5, 7]
  length =   [3, 2, 1]

  output = [b'hir', b'ee', b'n']
  ```

  Raises:

    * `ValueError`: If the first argument cannot be converted to a
       Tensor of `dtype string`.
    * `InvalidArgumentError`: If indices are out of range.
    * `ValueError`: If `pos` and `len` are not the same shape.

  Args:
    input: A `Tensor` of type `string`. Tensor of strings
    pos: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      Scalar defining the position of first character in each substring
    len: A `Tensor`. Must have the same type as `pos`.
      Scalar defining the number of characters to include in each substring
    unit: An optional `string` from: `"BYTE", "UTF8_CHAR"`. Defaults to `"BYTE"`.
      The unit that is used to create the substring.  One of: `"BYTE"` (for
      defining position and length by bytes) or `"UTF8_CHAR"` (for the UTF-8
      encoded Unicode code points).  The default is `"BYTE"`. Results are undefined if
      `unit=UTF8_CHAR` and the `input` strings do not contain structurally valid
      UTF-8.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `string`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "Substr", name, input, pos, len, "unit", unit)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return substr_eager_fallback(
          input, pos, len, unit=unit, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if unit is None:
    unit = "BYTE"
  unit = _execute.make_str(unit, "unit")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "Substr", input=input, pos=pos, len=len, unit=unit, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"), "unit", _op.get_attr("unit"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "Substr", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

Substr = tf_export("raw_ops.Substr")(_ops.to_raw_op(substr))


def substr_eager_fallback(input: Annotated[Any, _atypes.String], pos: Annotated[Any, TV_Substr_T], len: Annotated[Any, TV_Substr_T], unit: str, name, ctx) -> Annotated[Any, _atypes.String]:
  if unit is None:
    unit = "BYTE"
  unit = _execute.make_str(unit, "unit")
  _attr_T, _inputs_T = _execute.args_to_matching_eager([pos, len], ctx, [_dtypes.int32, _dtypes.int64, ])
  (pos, len) = _inputs_T
  input = _ops.convert_to_tensor(input, _dtypes.string)
  _inputs_flat = [input, pos, len]
  _attrs = ("T", _attr_T, "unit", unit)
  _result = _execute.execute(b"Substr", 1, inputs=_inputs_flat, attrs=_attrs,
                             ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "Substr", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

_UnicodeDecodeOutput = collections.namedtuple(
    "UnicodeDecode",
    ["row_splits", "char_values"])


TV_UnicodeDecode_Tsplits = TypeVar("TV_UnicodeDecode_Tsplits", _atypes.Int32, _atypes.Int64)

def unicode_decode(input: Annotated[Any, _atypes.String], input_encoding: str, errors:str="replace", replacement_char:int=65533, replace_control_characters:bool=False, Tsplits:TV_UnicodeDecode_Tsplits=_dtypes.int64, name=None):
  r"""Decodes each string in `input` into a sequence of Unicode code points.

  The character codepoints for all strings are returned using a single vector
  `char_values`, with strings expanded to characters in row-major order.

  The `row_splits` tensor indicates where the codepoints for
  each input string begin and end within the `char_values` tensor.
  In particular, the values for the `i`th
  string (in row-major order) are stored in the slice
  `[row_splits[i]:row_splits[i+1]]`. Thus:

  * `char_values[row_splits[i]+j]` is the Unicode codepoint for the `j`th
    character in the `i`th string (in row-major order).
  * `row_splits[i+1] - row_splits[i]` is the number of characters in the `i`th
    string (in row-major order).

  Args:
    input: A `Tensor` of type `string`.
      The text to be decoded. Can have any shape. Note that the output is flattened
      to a vector of char values.
    input_encoding: A `string`.
      Text encoding of the input strings. This is any of the encodings supported
      by ICU ucnv algorithmic converters. Examples: `"UTF-16", "US ASCII", "UTF-8"`.
    errors: An optional `string` from: `"strict", "replace", "ignore"`. Defaults to `"replace"`.
      Error handling policy when there is invalid formatting found in the input.
      The value of 'strict' will cause the operation to produce a InvalidArgument
      error on any invalid input formatting. A value of 'replace' (the default) will
      cause the operation to replace any invalid formatting in the input with the
      `replacement_char` codepoint. A value of 'ignore' will cause the operation to
      skip any invalid formatting in the input and produce no corresponding output
      character.
    replacement_char: An optional `int`. Defaults to `65533`.
      The replacement character codepoint to be used in place of any invalid
      formatting in the input when `errors='replace'`. Any valid unicode codepoint may
      be used. The default value is the default unicode replacement character is
      0xFFFD or U+65533.)
    replace_control_characters: An optional `bool`. Defaults to `False`.
      Whether to replace the C0 control characters (00-1F) with the
      `replacement_char`. Default is false.
    Tsplits: An optional `tf.DType` from: `tf.int32, tf.int64`. Defaults to `tf.int64`.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (row_splits, char_values).

    row_splits: A `Tensor` of type `Tsplits`.
    char_values: A `Tensor` of type `int32`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "UnicodeDecode", name, input, "input_encoding", input_encoding,
        "errors", errors, "replacement_char", replacement_char,
        "replace_control_characters", replace_control_characters, "Tsplits",
        Tsplits)
      _result = _UnicodeDecodeOutput._make(_result)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return unicode_decode_eager_fallback(
          input, input_encoding=input_encoding, errors=errors,
          replacement_char=replacement_char,
          replace_control_characters=replace_control_characters,
          Tsplits=Tsplits, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  input_encoding = _execute.make_str(input_encoding, "input_encoding")
  if errors is None:
    errors = "replace"
  errors = _execute.make_str(errors, "errors")
  if replacement_char is None:
    replacement_char = 65533
  replacement_char = _execute.make_int(replacement_char, "replacement_char")
  if replace_control_characters is None:
    replace_control_characters = False
  replace_control_characters = _execute.make_bool(replace_control_characters, "replace_control_characters")
  if Tsplits is None:
    Tsplits = _dtypes.int64
  Tsplits = _execute.make_type(Tsplits, "Tsplits")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "UnicodeDecode", input=input, input_encoding=input_encoding,
                         errors=errors, replacement_char=replacement_char,
                         replace_control_characters=replace_control_characters,
                         Tsplits=Tsplits, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("input_encoding", _op.get_attr("input_encoding"), "errors",
              _op.get_attr("errors"), "replacement_char",
              _op._get_attr_int("replacement_char"),
              "replace_control_characters",
              _op._get_attr_bool("replace_control_characters"), "Tsplits",
              _op._get_attr_type("Tsplits"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "UnicodeDecode", _inputs_flat, _attrs, _result)
  _result = _UnicodeDecodeOutput._make(_result)
  return _result

UnicodeDecode = tf_export("raw_ops.UnicodeDecode")(_ops.to_raw_op(unicode_decode))


def unicode_decode_eager_fallback(input: Annotated[Any, _atypes.String], input_encoding: str, errors: str, replacement_char: int, replace_control_characters: bool, Tsplits: TV_UnicodeDecode_Tsplits, name, ctx):
  input_encoding = _execute.make_str(input_encoding, "input_encoding")
  if errors is None:
    errors = "replace"
  errors = _execute.make_str(errors, "errors")
  if replacement_char is None:
    replacement_char = 65533
  replacement_char = _execute.make_int(replacement_char, "replacement_char")
  if replace_control_characters is None:
    replace_control_characters = False
  replace_control_characters = _execute.make_bool(replace_control_characters, "replace_control_characters")
  if Tsplits is None:
    Tsplits = _dtypes.int64
  Tsplits = _execute.make_type(Tsplits, "Tsplits")
  input = _ops.convert_to_tensor(input, _dtypes.string)
  _inputs_flat = [input]
  _attrs = ("input_encoding", input_encoding, "errors", errors,
  "replacement_char", replacement_char, "replace_control_characters",
  replace_control_characters, "Tsplits", Tsplits)
  _result = _execute.execute(b"UnicodeDecode", 2, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "UnicodeDecode", _inputs_flat, _attrs, _result)
  _result = _UnicodeDecodeOutput._make(_result)
  return _result

_UnicodeDecodeWithOffsetsOutput = collections.namedtuple(
    "UnicodeDecodeWithOffsets",
    ["row_splits", "char_values", "char_to_byte_starts"])


TV_UnicodeDecodeWithOffsets_Tsplits = TypeVar("TV_UnicodeDecodeWithOffsets_Tsplits", _atypes.Int32, _atypes.Int64)

def unicode_decode_with_offsets(input: Annotated[Any, _atypes.String], input_encoding: str, errors:str="replace", replacement_char:int=65533, replace_control_characters:bool=False, Tsplits:TV_UnicodeDecodeWithOffsets_Tsplits=_dtypes.int64, name=None):
  r"""Decodes each string in `input` into a sequence of Unicode code points.

  The character codepoints for all strings are returned using a single vector
  `char_values`, with strings expanded to characters in row-major order.
  Similarly, the character start byte offsets are returned using a single vector
  `char_to_byte_starts`, with strings expanded in row-major order.

  The `row_splits` tensor indicates where the codepoints and start offsets for
  each input string begin and end within the `char_values` and
  `char_to_byte_starts` tensors.  In particular, the values for the `i`th
  string (in row-major order) are stored in the slice
  `[row_splits[i]:row_splits[i+1]]`. Thus:

  * `char_values[row_splits[i]+j]` is the Unicode codepoint for the `j`th
    character in the `i`th string (in row-major order).
  * `char_to_bytes_starts[row_splits[i]+j]` is the start byte offset for the `j`th
    character in the `i`th string (in row-major order).
  * `row_splits[i+1] - row_splits[i]` is the number of characters in the `i`th
    string (in row-major order).

  Args:
    input: A `Tensor` of type `string`.
      The text to be decoded. Can have any shape. Note that the output is flattened
      to a vector of char values.
    input_encoding: A `string`.
      Text encoding of the input strings. This is any of the encodings supported
      by ICU ucnv algorithmic converters. Examples: `"UTF-16", "US ASCII", "UTF-8"`.
    errors: An optional `string` from: `"strict", "replace", "ignore"`. Defaults to `"replace"`.
      Error handling policy when there is invalid formatting found in the input.
      The value of 'strict' will cause the operation to produce a InvalidArgument
      error on any invalid input formatting. A value of 'replace' (the default) will
      cause the operation to replace any invalid formatting in the input with the
      `replacement_char` codepoint. A value of 'ignore' will cause the operation to
      skip any invalid formatting in the input and produce no corresponding output
      character.
    replacement_char: An optional `int`. Defaults to `65533`.
      The replacement character codepoint to be used in place of any invalid
      formatting in the input when `errors='replace'`. Any valid unicode codepoint may
      be used. The default value is the default unicode replacement character is
      0xFFFD or U+65533.)
    replace_control_characters: An optional `bool`. Defaults to `False`.
      Whether to replace the C0 control characters (00-1F) with the
      `replacement_char`. Default is false.
    Tsplits: An optional `tf.DType` from: `tf.int32, tf.int64`. Defaults to `tf.int64`.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (row_splits, char_values, char_to_byte_starts).

    row_splits: A `Tensor` of type `Tsplits`.
    char_values: A `Tensor` of type `int32`.
    char_to_byte_starts: A `Tensor` of type `int64`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "UnicodeDecodeWithOffsets", name, input, "input_encoding",
        input_encoding, "errors", errors, "replacement_char",
        replacement_char, "replace_control_characters",
        replace_control_characters, "Tsplits", Tsplits)
      _result = _UnicodeDecodeWithOffsetsOutput._make(_result)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return unicode_decode_with_offsets_eager_fallback(
          input, input_encoding=input_encoding, errors=errors,
          replacement_char=replacement_char,
          replace_control_characters=replace_control_characters,
          Tsplits=Tsplits, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  input_encoding = _execute.make_str(input_encoding, "input_encoding")
  if errors is None:
    errors = "replace"
  errors = _execute.make_str(errors, "errors")
  if replacement_char is None:
    replacement_char = 65533
  replacement_char = _execute.make_int(replacement_char, "replacement_char")
  if replace_control_characters is None:
    replace_control_characters = False
  replace_control_characters = _execute.make_bool(replace_control_characters, "replace_control_characters")
  if Tsplits is None:
    Tsplits = _dtypes.int64
  Tsplits = _execute.make_type(Tsplits, "Tsplits")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "UnicodeDecodeWithOffsets", input=input,
                                    input_encoding=input_encoding,
                                    errors=errors,
                                    replacement_char=replacement_char,
                                    replace_control_characters=replace_control_characters,
                                    Tsplits=Tsplits, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("input_encoding", _op.get_attr("input_encoding"), "errors",
              _op.get_attr("errors"), "replacement_char",
              _op._get_attr_int("replacement_char"),
              "replace_control_characters",
              _op._get_attr_bool("replace_control_characters"), "Tsplits",
              _op._get_attr_type("Tsplits"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "UnicodeDecodeWithOffsets", _inputs_flat, _attrs, _result)
  _result = _UnicodeDecodeWithOffsetsOutput._make(_result)
  return _result

UnicodeDecodeWithOffsets = tf_export("raw_ops.UnicodeDecodeWithOffsets")(_ops.to_raw_op(unicode_decode_with_offsets))


def unicode_decode_with_offsets_eager_fallback(input: Annotated[Any, _atypes.String], input_encoding: str, errors: str, replacement_char: int, replace_control_characters: bool, Tsplits: TV_UnicodeDecodeWithOffsets_Tsplits, name, ctx):
  input_encoding = _execute.make_str(input_encoding, "input_encoding")
  if errors is None:
    errors = "replace"
  errors = _execute.make_str(errors, "errors")
  if replacement_char is None:
    replacement_char = 65533
  replacement_char = _execute.make_int(replacement_char, "replacement_char")
  if replace_control_characters is None:
    replace_control_characters = False
  replace_control_characters = _execute.make_bool(replace_control_characters, "replace_control_characters")
  if Tsplits is None:
    Tsplits = _dtypes.int64
  Tsplits = _execute.make_type(Tsplits, "Tsplits")
  input = _ops.convert_to_tensor(input, _dtypes.string)
  _inputs_flat = [input]
  _attrs = ("input_encoding", input_encoding, "errors", errors,
  "replacement_char", replacement_char, "replace_control_characters",
  replace_control_characters, "Tsplits", Tsplits)
  _result = _execute.execute(b"UnicodeDecodeWithOffsets", 3,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "UnicodeDecodeWithOffsets", _inputs_flat, _attrs, _result)
  _result = _UnicodeDecodeWithOffsetsOutput._make(_result)
  return _result


TV_UnicodeEncode_Tsplits = TypeVar("TV_UnicodeEncode_Tsplits", _atypes.Int32, _atypes.Int64)

def unicode_encode(input_values: Annotated[Any, _atypes.Int32], input_splits: Annotated[Any, TV_UnicodeEncode_Tsplits], output_encoding: str, errors:str="replace", replacement_char:int=65533, name=None) -> Annotated[Any, _atypes.String]:
  r"""Encode a tensor of ints into unicode strings.

  Returns a vector of strings, where `output[i]` is constructed by encoding the
  Unicode codepoints in `input_values[input_splits[i]:input_splits[i+1]]`
  using `output_encoding`.

  ---

  Example:

  ```
  input_values = [72, 101, 108, 108, 111, 87, 111, 114, 108, 100]
  input_splits = [0, 5, 10]
  output_encoding = 'UTF-8'

  output = ['Hello', 'World']
  ```

  Args:
    input_values: A `Tensor` of type `int32`.
      A 1D tensor containing the unicode codepoints that should be encoded.
    input_splits: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      A 1D tensor specifying how the unicode codepoints should be split into strings.
      In particular, `output[i]` is constructed by encoding the codepoints in the
      slice `input_values[input_splits[i]:input_splits[i+1]]`.
    output_encoding: A `string` from: `"UTF-8", "UTF-16-BE", "UTF-32-BE"`.
      Unicode encoding of the output strings. Valid encodings are: `"UTF-8",
      "UTF-16-BE", and "UTF-32-BE"`.
    errors: An optional `string` from: `"ignore", "replace", "strict"`. Defaults to `"replace"`.
      Error handling policy when there is invalid formatting found in the input.
      The value of 'strict' will cause the operation to produce a InvalidArgument
      error on any invalid input formatting. A value of 'replace' (the default) will
      cause the operation to replace any invalid formatting in the input with the
      `replacement_char` codepoint. A value of 'ignore' will cause the operation to
      skip any invalid formatting in the input and produce no corresponding output
      character.
    replacement_char: An optional `int`. Defaults to `65533`.
      The replacement character codepoint to be used in place of any invalid
      formatting in the input when `errors='replace'`. Any valid unicode codepoint may
      be used. The default value is the default unicode replacement character is
      0xFFFD (U+65533).
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `string`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "UnicodeEncode", name, input_values, input_splits, "errors",
        errors, "output_encoding", output_encoding, "replacement_char",
        replacement_char)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return unicode_encode_eager_fallback(
          input_values, input_splits, errors=errors,
          output_encoding=output_encoding, replacement_char=replacement_char,
          name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  output_encoding = _execute.make_str(output_encoding, "output_encoding")
  if errors is None:
    errors = "replace"
  errors = _execute.make_str(errors, "errors")
  if replacement_char is None:
    replacement_char = 65533
  replacement_char = _execute.make_int(replacement_char, "replacement_char")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "UnicodeEncode", input_values=input_values, input_splits=input_splits,
                         output_encoding=output_encoding, errors=errors,
                         replacement_char=replacement_char, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("errors", _op.get_attr("errors"), "output_encoding",
              _op.get_attr("output_encoding"), "replacement_char",
              _op._get_attr_int("replacement_char"), "Tsplits",
              _op._get_attr_type("Tsplits"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "UnicodeEncode", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

UnicodeEncode = tf_export("raw_ops.UnicodeEncode")(_ops.to_raw_op(unicode_encode))


def unicode_encode_eager_fallback(input_values: Annotated[Any, _atypes.Int32], input_splits: Annotated[Any, TV_UnicodeEncode_Tsplits], output_encoding: str, errors: str, replacement_char: int, name, ctx) -> Annotated[Any, _atypes.String]:
  output_encoding = _execute.make_str(output_encoding, "output_encoding")
  if errors is None:
    errors = "replace"
  errors = _execute.make_str(errors, "errors")
  if replacement_char is None:
    replacement_char = 65533
  replacement_char = _execute.make_int(replacement_char, "replacement_char")
  _attr_Tsplits, (input_splits,) = _execute.args_to_matching_eager([input_splits], ctx, [_dtypes.int32, _dtypes.int64, ], _dtypes.int64)
  input_values = _ops.convert_to_tensor(input_values, _dtypes.int32)
  _inputs_flat = [input_values, input_splits]
  _attrs = ("errors", errors, "output_encoding", output_encoding,
  "replacement_char", replacement_char, "Tsplits", _attr_Tsplits)
  _result = _execute.execute(b"UnicodeEncode", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "UnicodeEncode", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('strings.unicode_script')
def unicode_script(input: Annotated[Any, _atypes.Int32], name=None) -> Annotated[Any, _atypes.Int32]:
  r"""Determine the script codes of a given tensor of Unicode integer code points.

  This operation converts Unicode code points to script codes corresponding to
  each code point. Script codes correspond to International Components for
  Unicode (ICU) UScriptCode values.

  See
  [ICU project docs](http://icu-project.org/apiref/icu4c/uscript_8h.html)
  for more details on script codes.

  For an example, see the unicode strings guide on [unicode scripts]
  (https://www.tensorflow.org/tutorials/load_data/unicode#representing_unicode).

  Returns -1 (USCRIPT_INVALID_CODE) for invalid codepoints. Output shape will
  match input shape.

  Examples:

  >>> tf.strings.unicode_script([1, 31, 38])
  <tf.Tensor: shape=(3,), dtype=int32, numpy=array([0, 0, 0], dtype=int32)>

  Args:
    input: A `Tensor` of type `int32`. A Tensor of int32 Unicode code points.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `int32`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "UnicodeScript", name, input)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_unicode_script(
          (input, name,), None)
      if _result is not NotImplemented:
        return _result
      return unicode_script_eager_fallback(
          input, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            unicode_script, (), dict(input=input, name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_unicode_script(
        (input, name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "UnicodeScript", input=input, name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          unicode_script, (), dict(input=input, name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ()
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "UnicodeScript", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

UnicodeScript = tf_export("raw_ops.UnicodeScript")(_ops.to_raw_op(unicode_script))
_dispatcher_for_unicode_script = unicode_script._tf_type_based_dispatcher.Dispatch


def unicode_script_eager_fallback(input: Annotated[Any, _atypes.Int32], name, ctx) -> Annotated[Any, _atypes.Int32]:
  input = _ops.convert_to_tensor(input, _dtypes.int32)
  _inputs_flat = [input]
  _attrs = None
  _result = _execute.execute(b"UnicodeScript", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "UnicodeScript", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('strings.unicode_transcode')
def unicode_transcode(input: Annotated[Any, _atypes.String], input_encoding: str, output_encoding: str, errors:str="replace", replacement_char:int=65533, replace_control_characters:bool=False, name=None) -> Annotated[Any, _atypes.String]:
  r"""Transcode the input text from a source encoding to a destination encoding.

  The input is a string tensor of any shape. The output is a string tensor of
  the same shape containing the transcoded strings. Output strings are always
  valid unicode. If the input contains invalid encoding positions, the
  `errors` attribute sets the policy for how to deal with them. If the default
  error-handling policy is used, invalid formatting will be substituted in the
  output by the `replacement_char`. If the errors policy is to `ignore`, any
  invalid encoding positions in the input are skipped and not included in the
  output. If it set to `strict` then any invalid formatting will result in an
  InvalidArgument error.

  This operation can be used with `output_encoding = input_encoding` to enforce
  correct formatting for inputs even if they are already in the desired encoding.

  If the input is prefixed by a Byte Order Mark needed to determine encoding
  (e.g. if the encoding is UTF-16 and the BOM indicates big-endian), then that
  BOM will be consumed and not emitted into the output. If the input encoding
  is marked with an explicit endianness (e.g. UTF-16-BE), then the BOM is
  interpreted as a non-breaking-space and is preserved in the output (including
  always for UTF-8).

  The end result is that if the input is marked as an explicit endianness the
  transcoding is faithful to all codepoints in the source. If it is not marked
  with an explicit endianness, the BOM is not considered part of the string itself
  but as metadata, and so is not preserved in the output.

  Examples:

  >>> tf.strings.unicode_transcode(["Hello", "TensorFlow", "2.x"], "UTF-8", "UTF-16-BE")
  <tf.Tensor: shape=(3,), dtype=string, numpy=
  array([b'\x00H\x00e\x00l\x00l\x00o',
         b'\x00T\x00e\x00n\x00s\x00o\x00r\x00F\x00l\x00o\x00w',
         b'\x002\x00.\x00x'], dtype=object)>
  >>> tf.strings.unicode_transcode(["A", "B", "C"], "US ASCII", "UTF-8").numpy()
  array([b'A', b'B', b'C'], dtype=object)

  Args:
    input: A `Tensor` of type `string`.
      The text to be processed. Can have any shape.
    input_encoding: A `string`.
      Text encoding of the input strings. This is any of the encodings supported
      by ICU ucnv algorithmic converters. Examples: `"UTF-16", "US ASCII", "UTF-8"`.
    output_encoding: A `string` from: `"UTF-8", "UTF-16-BE", "UTF-32-BE"`.
      The unicode encoding to use in the output. Must be one of
      `"UTF-8", "UTF-16-BE", "UTF-32-BE"`. Multi-byte encodings will be big-endian.
    errors: An optional `string` from: `"strict", "replace", "ignore"`. Defaults to `"replace"`.
      Error handling policy when there is invalid formatting found in the input.
      The value of 'strict' will cause the operation to produce a InvalidArgument
      error on any invalid input formatting. A value of 'replace' (the default) will
      cause the operation to replace any invalid formatting in the input with the
      `replacement_char` codepoint. A value of 'ignore' will cause the operation to
      skip any invalid formatting in the input and produce no corresponding output
      character.
    replacement_char: An optional `int`. Defaults to `65533`.
      The replacement character codepoint to be used in place of any invalid
      formatting in the input when `errors='replace'`. Any valid unicode codepoint may
      be used. The default value is the default unicode replacement character is
      0xFFFD or U+65533.)

      Note that for UTF-8, passing a replacement character expressible in 1 byte, such
      as ' ', will preserve string alignment to the source since invalid bytes will be
      replaced with a 1-byte replacement. For UTF-16-BE and UTF-16-LE, any 1 or 2 byte
      replacement character will preserve byte alignment to the source.
    replace_control_characters: An optional `bool`. Defaults to `False`.
      Whether to replace the C0 control characters (00-1F) with the
      `replacement_char`. Default is false.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `string`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "UnicodeTranscode", name, input, "input_encoding",
        input_encoding, "output_encoding", output_encoding, "errors", errors,
        "replacement_char", replacement_char, "replace_control_characters",
        replace_control_characters)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_unicode_transcode(
          (input, input_encoding, output_encoding, errors, replacement_char,
          replace_control_characters, name,), None)
      if _result is not NotImplemented:
        return _result
      return unicode_transcode_eager_fallback(
          input, input_encoding=input_encoding,
          output_encoding=output_encoding, errors=errors,
          replacement_char=replacement_char,
          replace_control_characters=replace_control_characters, name=name,
          ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            unicode_transcode, (), dict(input=input,
                                        input_encoding=input_encoding,
                                        output_encoding=output_encoding,
                                        errors=errors,
                                        replacement_char=replacement_char,
                                        replace_control_characters=replace_control_characters,
                                        name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_unicode_transcode(
        (input, input_encoding, output_encoding, errors, replacement_char,
        replace_control_characters, name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  input_encoding = _execute.make_str(input_encoding, "input_encoding")
  output_encoding = _execute.make_str(output_encoding, "output_encoding")
  if errors is None:
    errors = "replace"
  errors = _execute.make_str(errors, "errors")
  if replacement_char is None:
    replacement_char = 65533
  replacement_char = _execute.make_int(replacement_char, "replacement_char")
  if replace_control_characters is None:
    replace_control_characters = False
  replace_control_characters = _execute.make_bool(replace_control_characters, "replace_control_characters")
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "UnicodeTranscode", input=input, input_encoding=input_encoding,
                            output_encoding=output_encoding, errors=errors,
                            replacement_char=replacement_char,
                            replace_control_characters=replace_control_characters,
                            name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          unicode_transcode, (), dict(input=input,
                                      input_encoding=input_encoding,
                                      output_encoding=output_encoding,
                                      errors=errors,
                                      replacement_char=replacement_char,
                                      replace_control_characters=replace_control_characters,
                                      name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("input_encoding", _op.get_attr("input_encoding"),
              "output_encoding", _op.get_attr("output_encoding"), "errors",
              _op.get_attr("errors"), "replacement_char",
              _op._get_attr_int("replacement_char"),
              "replace_control_characters",
              _op._get_attr_bool("replace_control_characters"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "UnicodeTranscode", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

UnicodeTranscode = tf_export("raw_ops.UnicodeTranscode")(_ops.to_raw_op(unicode_transcode))
_dispatcher_for_unicode_transcode = unicode_transcode._tf_type_based_dispatcher.Dispatch


def unicode_transcode_eager_fallback(input: Annotated[Any, _atypes.String], input_encoding: str, output_encoding: str, errors: str, replacement_char: int, replace_control_characters: bool, name, ctx) -> Annotated[Any, _atypes.String]:
  input_encoding = _execute.make_str(input_encoding, "input_encoding")
  output_encoding = _execute.make_str(output_encoding, "output_encoding")
  if errors is None:
    errors = "replace"
  errors = _execute.make_str(errors, "errors")
  if replacement_char is None:
    replacement_char = 65533
  replacement_char = _execute.make_int(replacement_char, "replacement_char")
  if replace_control_characters is None:
    replace_control_characters = False
  replace_control_characters = _execute.make_bool(replace_control_characters, "replace_control_characters")
  input = _ops.convert_to_tensor(input, _dtypes.string)
  _inputs_flat = [input]
  _attrs = ("input_encoding", input_encoding, "output_encoding",
  output_encoding, "errors", errors, "replacement_char", replacement_char,
  "replace_control_characters", replace_control_characters)
  _result = _execute.execute(b"UnicodeTranscode", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "UnicodeTranscode", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_UnsortedSegmentJoin_Tindices = TypeVar("TV_UnsortedSegmentJoin_Tindices", _atypes.Int32, _atypes.Int64)
TV_UnsortedSegmentJoin_Tnumsegments = TypeVar("TV_UnsortedSegmentJoin_Tnumsegments", _atypes.Int32, _atypes.Int64)

def unsorted_segment_join(inputs: Annotated[Any, _atypes.String], segment_ids: Annotated[Any, TV_UnsortedSegmentJoin_Tindices], num_segments: Annotated[Any, TV_UnsortedSegmentJoin_Tnumsegments], separator:str="", name=None) -> Annotated[Any, _atypes.String]:
  r"""TODO: add doc.

  Args:
    inputs: A `Tensor` of type `string`.
    segment_ids: A `Tensor`. Must be one of the following types: `int32`, `int64`.
    num_segments: A `Tensor`. Must be one of the following types: `int32`, `int64`.
    separator: An optional `string`. Defaults to `""`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `string`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "UnsortedSegmentJoin", name, inputs, segment_ids, num_segments,
        "separator", separator)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return unsorted_segment_join_eager_fallback(
          inputs, segment_ids, num_segments, separator=separator, name=name,
          ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if separator is None:
    separator = ""
  separator = _execute.make_str(separator, "separator")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "UnsortedSegmentJoin", inputs=inputs, segment_ids=segment_ids,
                               num_segments=num_segments, separator=separator,
                               name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("separator", _op.get_attr("separator"), "Tindices",
              _op._get_attr_type("Tindices"), "Tnumsegments",
              _op._get_attr_type("Tnumsegments"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "UnsortedSegmentJoin", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

UnsortedSegmentJoin = tf_export("raw_ops.UnsortedSegmentJoin")(_ops.to_raw_op(unsorted_segment_join))


def unsorted_segment_join_eager_fallback(inputs: Annotated[Any, _atypes.String], segment_ids: Annotated[Any, TV_UnsortedSegmentJoin_Tindices], num_segments: Annotated[Any, TV_UnsortedSegmentJoin_Tnumsegments], separator: str, name, ctx) -> Annotated[Any, _atypes.String]:
  if separator is None:
    separator = ""
  separator = _execute.make_str(separator, "separator")
  _attr_Tindices, (segment_ids,) = _execute.args_to_matching_eager([segment_ids], ctx, [_dtypes.int32, _dtypes.int64, ])
  _attr_Tnumsegments, (num_segments,) = _execute.args_to_matching_eager([num_segments], ctx, [_dtypes.int32, _dtypes.int64, ], _dtypes.int32)
  inputs = _ops.convert_to_tensor(inputs, _dtypes.string)
  _inputs_flat = [inputs, segment_ids, num_segments]
  _attrs = ("separator", separator, "Tindices", _attr_Tindices,
  "Tnumsegments", _attr_Tnumsegments)
  _result = _execute.execute(b"UnsortedSegmentJoin", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "UnsortedSegmentJoin", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

