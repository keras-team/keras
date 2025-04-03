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

def _assert(condition: Annotated[Any, _atypes.Bool], data, summarize:int=3, name=None):
  r"""Asserts that the given condition is true.

  If `condition` evaluates to false, print the list of tensors in `data`.
  `summarize` determines how many entries of the tensors to print.

  Args:
    condition: A `Tensor` of type `bool`. The condition to evaluate.
    data: A list of `Tensor` objects.
      The tensors to print out when condition is false.
    summarize: An optional `int`. Defaults to `3`.
      Print this many entries of each tensor.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "Assert", name, condition, data, "summarize", summarize)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return _assert_eager_fallback(
          condition, data, summarize=summarize, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if summarize is None:
    summarize = 3
  summarize = _execute.make_int(summarize, "summarize")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "Assert", condition=condition, data=data, summarize=summarize,
                  name=name)
  return _op
Assert = tf_export("raw_ops.Assert")(_ops.to_raw_op(_assert))


def _assert_eager_fallback(condition: Annotated[Any, _atypes.Bool], data, summarize: int, name, ctx):
  if summarize is None:
    summarize = 3
  summarize = _execute.make_int(summarize, "summarize")
  _attr_T, data = _execute.convert_to_mixed_eager_tensors(data, ctx)
  condition = _ops.convert_to_tensor(condition, _dtypes.bool)
  _inputs_flat = [condition] + list(data)
  _attrs = ("T", _attr_T, "summarize", summarize)
  _result = _execute.execute(b"Assert", 0, inputs=_inputs_flat, attrs=_attrs,
                             ctx=ctx, name=name)
  _result = None
  return _result


def audio_summary(tag: Annotated[Any, _atypes.String], tensor: Annotated[Any, _atypes.Float32], sample_rate: float, max_outputs:int=3, name=None) -> Annotated[Any, _atypes.String]:
  r"""Outputs a `Summary` protocol buffer with audio.

  The summary has up to `max_outputs` summary values containing audio. The
  audio is built from `tensor` which must be 3-D with shape `[batch_size,
  frames, channels]` or 2-D with shape `[batch_size, frames]`. The values are
  assumed to be in the range of `[-1.0, 1.0]` with a sample rate of `sample_rate`.

  The `tag` argument is a scalar `Tensor` of type `string`.  It is used to
  build the `tag` of the summary values:

  *  If `max_outputs` is 1, the summary value tag is '*tag*/audio'.
  *  If `max_outputs` is greater than 1, the summary value tags are
     generated sequentially as '*tag*/audio/0', '*tag*/audio/1', etc.

  Args:
    tag: A `Tensor` of type `string`.
      Scalar. Used to build the `tag` attribute of the summary values.
    tensor: A `Tensor` of type `float32`. 2-D of shape `[batch_size, frames]`.
    sample_rate: A `float`. The sample rate of the signal in hertz.
    max_outputs: An optional `int` that is `>= 1`. Defaults to `3`.
      Max number of batch elements to generate audio for.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `string`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "AudioSummary", name, tag, tensor, "sample_rate", sample_rate,
        "max_outputs", max_outputs)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return audio_summary_eager_fallback(
          tag, tensor, sample_rate=sample_rate, max_outputs=max_outputs,
          name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  sample_rate = _execute.make_float(sample_rate, "sample_rate")
  if max_outputs is None:
    max_outputs = 3
  max_outputs = _execute.make_int(max_outputs, "max_outputs")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "AudioSummary", tag=tag, tensor=tensor, sample_rate=sample_rate,
                        max_outputs=max_outputs, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("sample_rate", _op.get_attr("sample_rate"), "max_outputs",
              _op._get_attr_int("max_outputs"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "AudioSummary", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

AudioSummary = tf_export("raw_ops.AudioSummary")(_ops.to_raw_op(audio_summary))


def audio_summary_eager_fallback(tag: Annotated[Any, _atypes.String], tensor: Annotated[Any, _atypes.Float32], sample_rate: float, max_outputs: int, name, ctx) -> Annotated[Any, _atypes.String]:
  sample_rate = _execute.make_float(sample_rate, "sample_rate")
  if max_outputs is None:
    max_outputs = 3
  max_outputs = _execute.make_int(max_outputs, "max_outputs")
  tag = _ops.convert_to_tensor(tag, _dtypes.string)
  tensor = _ops.convert_to_tensor(tensor, _dtypes.float32)
  _inputs_flat = [tag, tensor]
  _attrs = ("sample_rate", sample_rate, "max_outputs", max_outputs)
  _result = _execute.execute(b"AudioSummary", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "AudioSummary", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


def audio_summary_v2(tag: Annotated[Any, _atypes.String], tensor: Annotated[Any, _atypes.Float32], sample_rate: Annotated[Any, _atypes.Float32], max_outputs:int=3, name=None) -> Annotated[Any, _atypes.String]:
  r"""Outputs a `Summary` protocol buffer with audio.

  The summary has up to `max_outputs` summary values containing audio. The
  audio is built from `tensor` which must be 3-D with shape `[batch_size,
  frames, channels]` or 2-D with shape `[batch_size, frames]`. The values are
  assumed to be in the range of `[-1.0, 1.0]` with a sample rate of `sample_rate`.

  The `tag` argument is a scalar `Tensor` of type `string`.  It is used to
  build the `tag` of the summary values:

  *  If `max_outputs` is 1, the summary value tag is '*tag*/audio'.
  *  If `max_outputs` is greater than 1, the summary value tags are
     generated sequentially as '*tag*/audio/0', '*tag*/audio/1', etc.

  Args:
    tag: A `Tensor` of type `string`.
      Scalar. Used to build the `tag` attribute of the summary values.
    tensor: A `Tensor` of type `float32`. 2-D of shape `[batch_size, frames]`.
    sample_rate: A `Tensor` of type `float32`.
      The sample rate of the signal in hertz.
    max_outputs: An optional `int` that is `>= 1`. Defaults to `3`.
      Max number of batch elements to generate audio for.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `string`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "AudioSummaryV2", name, tag, tensor, sample_rate, "max_outputs",
        max_outputs)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return audio_summary_v2_eager_fallback(
          tag, tensor, sample_rate, max_outputs=max_outputs, name=name,
          ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if max_outputs is None:
    max_outputs = 3
  max_outputs = _execute.make_int(max_outputs, "max_outputs")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "AudioSummaryV2", tag=tag, tensor=tensor, sample_rate=sample_rate,
                          max_outputs=max_outputs, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("max_outputs", _op._get_attr_int("max_outputs"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "AudioSummaryV2", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

AudioSummaryV2 = tf_export("raw_ops.AudioSummaryV2")(_ops.to_raw_op(audio_summary_v2))


def audio_summary_v2_eager_fallback(tag: Annotated[Any, _atypes.String], tensor: Annotated[Any, _atypes.Float32], sample_rate: Annotated[Any, _atypes.Float32], max_outputs: int, name, ctx) -> Annotated[Any, _atypes.String]:
  if max_outputs is None:
    max_outputs = 3
  max_outputs = _execute.make_int(max_outputs, "max_outputs")
  tag = _ops.convert_to_tensor(tag, _dtypes.string)
  tensor = _ops.convert_to_tensor(tensor, _dtypes.float32)
  sample_rate = _ops.convert_to_tensor(sample_rate, _dtypes.float32)
  _inputs_flat = [tag, tensor, sample_rate]
  _attrs = ("max_outputs", max_outputs)
  _result = _execute.execute(b"AudioSummaryV2", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "AudioSummaryV2", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_HistogramSummary_T = TypeVar("TV_HistogramSummary_T", _atypes.BFloat16, _atypes.Float32, _atypes.Float64, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int64, _atypes.Int8, _atypes.UInt16, _atypes.UInt32, _atypes.UInt64, _atypes.UInt8)

def histogram_summary(tag: Annotated[Any, _atypes.String], values: Annotated[Any, TV_HistogramSummary_T], name=None) -> Annotated[Any, _atypes.String]:
  r"""Outputs a `Summary` protocol buffer with a histogram.

  The generated
  [`Summary`](https://www.tensorflow.org/code/tensorflow/core/framework/summary.proto)
  has one summary value containing a histogram for `values`.

  This op reports an `InvalidArgument` error if any value is not finite.

  Args:
    tag: A `Tensor` of type `string`.
      Scalar.  Tag to use for the `Summary.Value`.
    values: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`, `int64`, `bfloat16`, `uint16`, `half`, `uint32`, `uint64`.
      Any shape. Values to use to build the histogram.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `string`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "HistogramSummary", name, tag, values)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return histogram_summary_eager_fallback(
          tag, values, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "HistogramSummary", tag=tag, values=values, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "HistogramSummary", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

HistogramSummary = tf_export("raw_ops.HistogramSummary")(_ops.to_raw_op(histogram_summary))


def histogram_summary_eager_fallback(tag: Annotated[Any, _atypes.String], values: Annotated[Any, TV_HistogramSummary_T], name, ctx) -> Annotated[Any, _atypes.String]:
  _attr_T, (values,) = _execute.args_to_matching_eager([values], ctx, [_dtypes.float32, _dtypes.float64, _dtypes.int32, _dtypes.uint8, _dtypes.int16, _dtypes.int8, _dtypes.int64, _dtypes.bfloat16, _dtypes.uint16, _dtypes.half, _dtypes.uint32, _dtypes.uint64, ], _dtypes.float32)
  tag = _ops.convert_to_tensor(tag, _dtypes.string)
  _inputs_flat = [tag, values]
  _attrs = ("T", _attr_T)
  _result = _execute.execute(b"HistogramSummary", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "HistogramSummary", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_ImageSummary_T = TypeVar("TV_ImageSummary_T", _atypes.Float32, _atypes.Float64, _atypes.Half, _atypes.UInt8)

def image_summary(tag: Annotated[Any, _atypes.String], tensor: Annotated[Any, TV_ImageSummary_T], max_images:int=3, bad_color=_execute.make_tensor("""dtype: DT_UINT8 tensor_shape { dim { size: 4 } } int_val: 255 int_val: 0 int_val: 0 int_val: 255 """, "bad_color"), name=None) -> Annotated[Any, _atypes.String]:
  r"""Outputs a `Summary` protocol buffer with images.

  The summary has up to `max_images` summary values containing images. The
  images are built from `tensor` which must be 4-D with shape `[batch_size,
  height, width, channels]` and where `channels` can be:

  *  1: `tensor` is interpreted as Grayscale.
  *  3: `tensor` is interpreted as RGB.
  *  4: `tensor` is interpreted as RGBA.

  The images have the same number of channels as the input tensor. For float
  input, the values are normalized one image at a time to fit in the range
  `[0, 255]`.  `uint8` values are unchanged.  The op uses two different
  normalization algorithms:

  *  If the input values are all positive, they are rescaled so the largest one
     is 255.

  *  If any input value is negative, the values are shifted so input value 0.0
     is at 127.  They are then rescaled so that either the smallest value is 0,
     or the largest one is 255.

  The `tag` argument is a scalar `Tensor` of type `string`.  It is used to
  build the `tag` of the summary values:

  *  If `max_images` is 1, the summary value tag is '*tag*/image'.
  *  If `max_images` is greater than 1, the summary value tags are
     generated sequentially as '*tag*/image/0', '*tag*/image/1', etc.

  The `bad_color` argument is the color to use in the generated images for
  non-finite input values.  It is a `uint8` 1-D tensor of length `channels`.
  Each element must be in the range `[0, 255]` (It represents the value of a
  pixel in the output image).  Non-finite values in the input tensor are
  replaced by this tensor in the output image.  The default value is the color
  red.

  Args:
    tag: A `Tensor` of type `string`.
      Scalar. Used to build the `tag` attribute of the summary values.
    tensor: A `Tensor`. Must be one of the following types: `uint8`, `float32`, `half`, `float64`.
      4-D of shape `[batch_size, height, width, channels]` where
      `channels` is 1, 3, or 4.
    max_images: An optional `int` that is `>= 1`. Defaults to `3`.
      Max number of batch elements to generate images for.
    bad_color: An optional `tf.TensorProto`. Defaults to `dtype: DT_UINT8 tensor_shape { dim { size: 4 } } int_val: 255 int_val: 0 int_val: 0 int_val: 255`.
      Color to use for pixels with non-finite values.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `string`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "ImageSummary", name, tag, tensor, "max_images", max_images,
        "bad_color", bad_color)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return image_summary_eager_fallback(
          tag, tensor, max_images=max_images, bad_color=bad_color, name=name,
          ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if max_images is None:
    max_images = 3
  max_images = _execute.make_int(max_images, "max_images")
  if bad_color is None:
    bad_color = _execute.make_tensor("""dtype: DT_UINT8 tensor_shape { dim { size: 4 } } int_val: 255 int_val: 0 int_val: 0 int_val: 255 """, "bad_color")
  bad_color = _execute.make_tensor(bad_color, "bad_color")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "ImageSummary", tag=tag, tensor=tensor, max_images=max_images,
                        bad_color=bad_color, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("max_images", _op._get_attr_int("max_images"), "T",
              _op._get_attr_type("T"), "bad_color", _op.get_attr("bad_color"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "ImageSummary", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

ImageSummary = tf_export("raw_ops.ImageSummary")(_ops.to_raw_op(image_summary))


def image_summary_eager_fallback(tag: Annotated[Any, _atypes.String], tensor: Annotated[Any, TV_ImageSummary_T], max_images: int, bad_color, name, ctx) -> Annotated[Any, _atypes.String]:
  if max_images is None:
    max_images = 3
  max_images = _execute.make_int(max_images, "max_images")
  if bad_color is None:
    bad_color = _execute.make_tensor("""dtype: DT_UINT8 tensor_shape { dim { size: 4 } } int_val: 255 int_val: 0 int_val: 0 int_val: 255 """, "bad_color")
  bad_color = _execute.make_tensor(bad_color, "bad_color")
  _attr_T, (tensor,) = _execute.args_to_matching_eager([tensor], ctx, [_dtypes.uint8, _dtypes.float32, _dtypes.half, _dtypes.float64, ], _dtypes.float32)
  tag = _ops.convert_to_tensor(tag, _dtypes.string)
  _inputs_flat = [tag, tensor]
  _attrs = ("max_images", max_images, "T", _attr_T, "bad_color", bad_color)
  _result = _execute.execute(b"ImageSummary", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "ImageSummary", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


def merge_summary(inputs: Annotated[List[Any], _atypes.String], name=None) -> Annotated[Any, _atypes.String]:
  r"""Merges summaries.

  This op creates a
  [`Summary`](https://www.tensorflow.org/code/tensorflow/core/framework/summary.proto)
  protocol buffer that contains the union of all the values in the input
  summaries.

  When the Op is run, it reports an `InvalidArgument` error if multiple values
  in the summaries to merge use the same tag.

  Args:
    inputs: A list of at least 1 `Tensor` objects with type `string`.
      Can be of any shape.  Each must contain serialized `Summary` protocol
      buffers.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `string`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "MergeSummary", name, inputs)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return merge_summary_eager_fallback(
          inputs, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if not isinstance(inputs, (list, tuple)):
    raise TypeError(
        "Expected list for 'inputs' argument to "
        "'merge_summary' Op, not %r." % inputs)
  _attr_N = len(inputs)
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "MergeSummary", inputs=inputs, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("N", _op._get_attr_int("N"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "MergeSummary", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

MergeSummary = tf_export("raw_ops.MergeSummary")(_ops.to_raw_op(merge_summary))


def merge_summary_eager_fallback(inputs: Annotated[List[Any], _atypes.String], name, ctx) -> Annotated[Any, _atypes.String]:
  if not isinstance(inputs, (list, tuple)):
    raise TypeError(
        "Expected list for 'inputs' argument to "
        "'merge_summary' Op, not %r." % inputs)
  _attr_N = len(inputs)
  inputs = _ops.convert_n_to_tensor(inputs, _dtypes.string)
  _inputs_flat = list(inputs)
  _attrs = ("N", _attr_N)
  _result = _execute.execute(b"MergeSummary", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "MergeSummary", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_Print_T = TypeVar("TV_Print_T", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float16, _atypes.Float32, _atypes.Float64, _atypes.Float8e4m3fn, _atypes.Float8e5m2, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int4, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.Resource, _atypes.String, _atypes.UInt16, _atypes.UInt32, _atypes.UInt4, _atypes.UInt64, _atypes.UInt8, _atypes.Variant)

def _print(input: Annotated[Any, TV_Print_T], data, message:str="", first_n:int=-1, summarize:int=3, name=None) -> Annotated[Any, TV_Print_T]:
  r"""Prints a list of tensors.

  Passes `input` through to `output` and prints `data` when evaluating.

  Args:
    input: A `Tensor`. The tensor passed to `output`
    data: A list of `Tensor` objects.
      A list of tensors to print out when op is evaluated.
    message: An optional `string`. Defaults to `""`.
      A string, prefix of the error message.
    first_n: An optional `int`. Defaults to `-1`.
      Only log `first_n` number of times. -1 disables logging.
    summarize: An optional `int`. Defaults to `3`.
      Only print this many entries of each tensor.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "Print", name, input, data, "message", message, "first_n",
        first_n, "summarize", summarize)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return _print_eager_fallback(
          input, data, message=message, first_n=first_n, summarize=summarize,
          name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if message is None:
    message = ""
  message = _execute.make_str(message, "message")
  if first_n is None:
    first_n = -1
  first_n = _execute.make_int(first_n, "first_n")
  if summarize is None:
    summarize = 3
  summarize = _execute.make_int(summarize, "summarize")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "Print", input=input, data=data, message=message, first_n=first_n,
                 summarize=summarize, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"), "U", _op.get_attr("U"), "message",
              _op.get_attr("message"), "first_n",
              _op._get_attr_int("first_n"), "summarize",
              _op._get_attr_int("summarize"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "Print", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

Print = tf_export("raw_ops.Print")(_ops.to_raw_op(_print))


def _print_eager_fallback(input: Annotated[Any, TV_Print_T], data, message: str, first_n: int, summarize: int, name, ctx) -> Annotated[Any, TV_Print_T]:
  if message is None:
    message = ""
  message = _execute.make_str(message, "message")
  if first_n is None:
    first_n = -1
  first_n = _execute.make_int(first_n, "first_n")
  if summarize is None:
    summarize = 3
  summarize = _execute.make_int(summarize, "summarize")
  _attr_T, (input,) = _execute.args_to_matching_eager([input], ctx, [])
  _attr_U, data = _execute.convert_to_mixed_eager_tensors(data, ctx)
  _inputs_flat = [input] + list(data)
  _attrs = ("T", _attr_T, "U", _attr_U, "message", message, "first_n",
  first_n, "summarize", summarize)
  _result = _execute.execute(b"Print", 1, inputs=_inputs_flat, attrs=_attrs,
                             ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "Print", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


def print_v2(input: Annotated[Any, _atypes.String], output_stream:str="stderr", end:str="\n", name=None):
  r"""Prints a string scalar.

  Prints a string scalar to the desired output_stream.

  Args:
    input: A `Tensor` of type `string`. The string scalar to print.
    output_stream: An optional `string`. Defaults to `"stderr"`.
      A string specifying the output stream or logging level to print to.
    end: An optional `string`. Defaults to `"\n"`.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "PrintV2", name, input, "output_stream", output_stream, "end",
        end)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return print_v2_eager_fallback(
          input, output_stream=output_stream, end=end, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if output_stream is None:
    output_stream = "stderr"
  output_stream = _execute.make_str(output_stream, "output_stream")
  if end is None:
    end = "\n"
  end = _execute.make_str(end, "end")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "PrintV2", input=input, output_stream=output_stream, end=end,
                   name=name)
  return _op
PrintV2 = tf_export("raw_ops.PrintV2")(_ops.to_raw_op(print_v2))


def print_v2_eager_fallback(input: Annotated[Any, _atypes.String], output_stream: str, end: str, name, ctx):
  if output_stream is None:
    output_stream = "stderr"
  output_stream = _execute.make_str(output_stream, "output_stream")
  if end is None:
    end = "\n"
  end = _execute.make_str(end, "end")
  input = _ops.convert_to_tensor(input, _dtypes.string)
  _inputs_flat = [input]
  _attrs = ("output_stream", output_stream, "end", end)
  _result = _execute.execute(b"PrintV2", 0, inputs=_inputs_flat, attrs=_attrs,
                             ctx=ctx, name=name)
  _result = None
  return _result


TV_ScalarSummary_T = TypeVar("TV_ScalarSummary_T", _atypes.BFloat16, _atypes.Float32, _atypes.Float64, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int64, _atypes.Int8, _atypes.UInt16, _atypes.UInt32, _atypes.UInt64, _atypes.UInt8)

def scalar_summary(tags: Annotated[Any, _atypes.String], values: Annotated[Any, TV_ScalarSummary_T], name=None) -> Annotated[Any, _atypes.String]:
  r"""Outputs a `Summary` protocol buffer with scalar values.

  The input `tags` and `values` must have the same shape.  The generated summary
  has a summary value for each tag-value pair in `tags` and `values`.

  Args:
    tags: A `Tensor` of type `string`. Tags for the summary.
    values: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`, `int64`, `bfloat16`, `uint16`, `half`, `uint32`, `uint64`.
      Same shape as `tags.  Values for the summary.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `string`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "ScalarSummary", name, tags, values)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return scalar_summary_eager_fallback(
          tags, values, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "ScalarSummary", tags=tags, values=values, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "ScalarSummary", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

ScalarSummary = tf_export("raw_ops.ScalarSummary")(_ops.to_raw_op(scalar_summary))


def scalar_summary_eager_fallback(tags: Annotated[Any, _atypes.String], values: Annotated[Any, TV_ScalarSummary_T], name, ctx) -> Annotated[Any, _atypes.String]:
  _attr_T, (values,) = _execute.args_to_matching_eager([values], ctx, [_dtypes.float32, _dtypes.float64, _dtypes.int32, _dtypes.uint8, _dtypes.int16, _dtypes.int8, _dtypes.int64, _dtypes.bfloat16, _dtypes.uint16, _dtypes.half, _dtypes.uint32, _dtypes.uint64, ])
  tags = _ops.convert_to_tensor(tags, _dtypes.string)
  _inputs_flat = [tags, values]
  _attrs = ("T", _attr_T)
  _result = _execute.execute(b"ScalarSummary", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "ScalarSummary", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_TensorSummary_T = TypeVar("TV_TensorSummary_T", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float16, _atypes.Float32, _atypes.Float64, _atypes.Float8e4m3fn, _atypes.Float8e5m2, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int4, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.Resource, _atypes.String, _atypes.UInt16, _atypes.UInt32, _atypes.UInt4, _atypes.UInt64, _atypes.UInt8, _atypes.Variant)

def tensor_summary(tensor: Annotated[Any, TV_TensorSummary_T], description:str="", labels=[], display_name:str="", name=None) -> Annotated[Any, _atypes.String]:
  r"""Outputs a `Summary` protocol buffer with a tensor.

  This op is being phased out in favor of TensorSummaryV2, which lets callers pass
  a tag as well as a serialized SummaryMetadata proto string that contains
  plugin-specific data. We will keep this op to maintain backwards compatibility.

  Args:
    tensor: A `Tensor`. A tensor to serialize.
    description: An optional `string`. Defaults to `""`.
      A json-encoded SummaryDescription proto.
    labels: An optional list of `strings`. Defaults to `[]`.
      An unused list of strings.
    display_name: An optional `string`. Defaults to `""`. An unused string.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `string`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "TensorSummary", name, tensor, "description", description,
        "labels", labels, "display_name", display_name)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return tensor_summary_eager_fallback(
          tensor, description=description, labels=labels,
          display_name=display_name, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if description is None:
    description = ""
  description = _execute.make_str(description, "description")
  if labels is None:
    labels = []
  if not isinstance(labels, (list, tuple)):
    raise TypeError(
        "Expected list for 'labels' argument to "
        "'tensor_summary' Op, not %r." % labels)
  labels = [_execute.make_str(_s, "labels") for _s in labels]
  if display_name is None:
    display_name = ""
  display_name = _execute.make_str(display_name, "display_name")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "TensorSummary", tensor=tensor, description=description,
                         labels=labels, display_name=display_name, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"), "description",
              _op.get_attr("description"), "labels", _op.get_attr("labels"),
              "display_name", _op.get_attr("display_name"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "TensorSummary", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

TensorSummary = tf_export("raw_ops.TensorSummary")(_ops.to_raw_op(tensor_summary))


def tensor_summary_eager_fallback(tensor: Annotated[Any, TV_TensorSummary_T], description: str, labels, display_name: str, name, ctx) -> Annotated[Any, _atypes.String]:
  if description is None:
    description = ""
  description = _execute.make_str(description, "description")
  if labels is None:
    labels = []
  if not isinstance(labels, (list, tuple)):
    raise TypeError(
        "Expected list for 'labels' argument to "
        "'tensor_summary' Op, not %r." % labels)
  labels = [_execute.make_str(_s, "labels") for _s in labels]
  if display_name is None:
    display_name = ""
  display_name = _execute.make_str(display_name, "display_name")
  _attr_T, (tensor,) = _execute.args_to_matching_eager([tensor], ctx, [])
  _inputs_flat = [tensor]
  _attrs = ("T", _attr_T, "description", description, "labels", labels,
  "display_name", display_name)
  _result = _execute.execute(b"TensorSummary", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "TensorSummary", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_TensorSummaryV2_T = TypeVar("TV_TensorSummaryV2_T", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float16, _atypes.Float32, _atypes.Float64, _atypes.Float8e4m3fn, _atypes.Float8e5m2, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int4, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.Resource, _atypes.String, _atypes.UInt16, _atypes.UInt32, _atypes.UInt4, _atypes.UInt64, _atypes.UInt8, _atypes.Variant)

def tensor_summary_v2(tag: Annotated[Any, _atypes.String], tensor: Annotated[Any, TV_TensorSummaryV2_T], serialized_summary_metadata: Annotated[Any, _atypes.String], name=None) -> Annotated[Any, _atypes.String]:
  r"""Outputs a `Summary` protocol buffer with a tensor and per-plugin data.

  Args:
    tag: A `Tensor` of type `string`.
      A string attached to this summary. Used for organization in TensorBoard.
    tensor: A `Tensor`. A tensor to serialize.
    serialized_summary_metadata: A `Tensor` of type `string`.
      A serialized SummaryMetadata proto. Contains plugin
      data.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `string`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "TensorSummaryV2", name, tag, tensor,
        serialized_summary_metadata)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return tensor_summary_v2_eager_fallback(
          tag, tensor, serialized_summary_metadata, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "TensorSummaryV2", tag=tag, tensor=tensor,
                           serialized_summary_metadata=serialized_summary_metadata,
                           name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "TensorSummaryV2", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

TensorSummaryV2 = tf_export("raw_ops.TensorSummaryV2")(_ops.to_raw_op(tensor_summary_v2))


def tensor_summary_v2_eager_fallback(tag: Annotated[Any, _atypes.String], tensor: Annotated[Any, TV_TensorSummaryV2_T], serialized_summary_metadata: Annotated[Any, _atypes.String], name, ctx) -> Annotated[Any, _atypes.String]:
  _attr_T, (tensor,) = _execute.args_to_matching_eager([tensor], ctx, [])
  tag = _ops.convert_to_tensor(tag, _dtypes.string)
  serialized_summary_metadata = _ops.convert_to_tensor(serialized_summary_metadata, _dtypes.string)
  _inputs_flat = [tag, tensor, serialized_summary_metadata]
  _attrs = ("T", _attr_T)
  _result = _execute.execute(b"TensorSummaryV2", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "TensorSummaryV2", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('timestamp')
def timestamp(name=None) -> Annotated[Any, _atypes.Float64]:
  r"""Provides the time since epoch in seconds.

  Returns the timestamp as a `float64` for seconds since the Unix epoch.

  Common usages include:
  * Logging
  * Providing a random number seed
  * Debugging graph execution
  * Generating timing information, mainly through comparison of timestamps

  Note: In graph mode, the timestamp is computed when the op is executed,
  not when it is added to the graph.  In eager mode, the timestamp is computed
  when the op is eagerly executed.

  Args:
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `float64`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "Timestamp", name)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_timestamp(
          (name,), None)
      if _result is not NotImplemented:
        return _result
      return timestamp_eager_fallback(
          name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            timestamp, (), dict(name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_timestamp(
        (name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "Timestamp", name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          timestamp, (), dict(name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ()
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "Timestamp", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

Timestamp = tf_export("raw_ops.Timestamp")(_ops.to_raw_op(timestamp))
_dispatcher_for_timestamp = timestamp._tf_type_based_dispatcher.Dispatch


def timestamp_eager_fallback(name, ctx) -> Annotated[Any, _atypes.Float64]:
  _inputs_flat = []
  _attrs = None
  _result = _execute.execute(b"Timestamp", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "Timestamp", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

