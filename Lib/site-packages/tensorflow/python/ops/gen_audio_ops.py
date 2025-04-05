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

def audio_spectrogram(input: Annotated[Any, _atypes.Float32], window_size: int, stride: int, magnitude_squared:bool=False, name=None) -> Annotated[Any, _atypes.Float32]:
  r"""Produces a visualization of audio data over time.

  Spectrograms are a standard way of representing audio information as a series of
  slices of frequency information, one slice for each window of time. By joining
  these together into a sequence, they form a distinctive fingerprint of the sound
  over time.

  This op expects to receive audio data as an input, stored as floats in the range
  -1 to 1, together with a window width in samples, and a stride specifying how
  far to move the window between slices. From this it generates a three
  dimensional output. The first dimension is for the channels in the input, so a
  stereo audio input would have two here for example. The second dimension is time,
  with successive frequency slices. The third dimension has an amplitude value for
  each frequency during that time slice.

  This means the layout when converted and saved as an image is rotated 90 degrees
  clockwise from a typical spectrogram. Time is descending down the Y axis, and
  the frequency decreases from left to right.

  Each value in the result represents the square root of the sum of the real and
  imaginary parts of an FFT on the current window of samples. In this way, the
  lowest dimension represents the power of each frequency in the current window,
  and adjacent windows are concatenated in the next dimension.

  To get a more intuitive and visual look at what this operation does, you can run
  tensorflow/examples/wav_to_spectrogram to read in an audio file and save out the
  resulting spectrogram as a PNG image.

  Args:
    input: A `Tensor` of type `float32`. Float representation of audio data.
    window_size: An `int`.
      How wide the input window is in samples. For the highest efficiency
      this should be a power of two, but other values are accepted.
    stride: An `int`.
      How widely apart the center of adjacent sample windows should be.
    magnitude_squared: An optional `bool`. Defaults to `False`.
      Whether to return the squared magnitude or just the
      magnitude. Using squared magnitude can avoid extra calculations.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `float32`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "AudioSpectrogram", name, input, "window_size", window_size,
        "stride", stride, "magnitude_squared", magnitude_squared)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return audio_spectrogram_eager_fallback(
          input, window_size=window_size, stride=stride,
          magnitude_squared=magnitude_squared, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  window_size = _execute.make_int(window_size, "window_size")
  stride = _execute.make_int(stride, "stride")
  if magnitude_squared is None:
    magnitude_squared = False
  magnitude_squared = _execute.make_bool(magnitude_squared, "magnitude_squared")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "AudioSpectrogram", input=input, window_size=window_size,
                            stride=stride,
                            magnitude_squared=magnitude_squared, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("window_size", _op._get_attr_int("window_size"), "stride",
              _op._get_attr_int("stride"), "magnitude_squared",
              _op._get_attr_bool("magnitude_squared"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "AudioSpectrogram", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

AudioSpectrogram = tf_export("raw_ops.AudioSpectrogram")(_ops.to_raw_op(audio_spectrogram))


def audio_spectrogram_eager_fallback(input: Annotated[Any, _atypes.Float32], window_size: int, stride: int, magnitude_squared: bool, name, ctx) -> Annotated[Any, _atypes.Float32]:
  window_size = _execute.make_int(window_size, "window_size")
  stride = _execute.make_int(stride, "stride")
  if magnitude_squared is None:
    magnitude_squared = False
  magnitude_squared = _execute.make_bool(magnitude_squared, "magnitude_squared")
  input = _ops.convert_to_tensor(input, _dtypes.float32)
  _inputs_flat = [input]
  _attrs = ("window_size", window_size, "stride", stride, "magnitude_squared",
  magnitude_squared)
  _result = _execute.execute(b"AudioSpectrogram", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "AudioSpectrogram", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

_DecodeWavOutput = collections.namedtuple(
    "DecodeWav",
    ["audio", "sample_rate"])


@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('audio.decode_wav')
def decode_wav(contents: Annotated[Any, _atypes.String], desired_channels:int=-1, desired_samples:int=-1, name=None):
  r"""Decode a 16-bit PCM WAV file to a float tensor.

  The -32768 to 32767 signed 16-bit values will be scaled to -1.0 to 1.0 in float.

  When desired_channels is set, if the input contains fewer channels than this
  then the last channel will be duplicated to give the requested number, else if
  the input has more channels than requested then the additional channels will be
  ignored.

  If desired_samples is set, then the audio will be cropped or padded with zeroes
  to the requested length.

  The first output contains a Tensor with the content of the audio samples. The
  lowest dimension will be the number of channels, and the second will be the
  number of samples. For example, a ten-sample-long stereo WAV file should give an
  output shape of [10, 2].

  Args:
    contents: A `Tensor` of type `string`.
      The WAV-encoded audio, usually from a file.
    desired_channels: An optional `int`. Defaults to `-1`.
      Number of sample channels wanted.
    desired_samples: An optional `int`. Defaults to `-1`.
      Length of audio requested.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (audio, sample_rate).

    audio: A `Tensor` of type `float32`.
    sample_rate: A `Tensor` of type `int32`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "DecodeWav", name, contents, "desired_channels",
        desired_channels, "desired_samples", desired_samples)
      _result = _DecodeWavOutput._make(_result)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_decode_wav(
          (contents, desired_channels, desired_samples, name,), None)
      if _result is not NotImplemented:
        return _result
      return decode_wav_eager_fallback(
          contents, desired_channels=desired_channels,
          desired_samples=desired_samples, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            decode_wav, (), dict(contents=contents,
                                 desired_channels=desired_channels,
                                 desired_samples=desired_samples, name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_decode_wav(
        (contents, desired_channels, desired_samples, name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  if desired_channels is None:
    desired_channels = -1
  desired_channels = _execute.make_int(desired_channels, "desired_channels")
  if desired_samples is None:
    desired_samples = -1
  desired_samples = _execute.make_int(desired_samples, "desired_samples")
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "DecodeWav", contents=contents, desired_channels=desired_channels,
                     desired_samples=desired_samples, name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          decode_wav, (), dict(contents=contents,
                               desired_channels=desired_channels,
                               desired_samples=desired_samples, name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("desired_channels", _op._get_attr_int("desired_channels"),
              "desired_samples", _op._get_attr_int("desired_samples"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "DecodeWav", _inputs_flat, _attrs, _result)
  _result = _DecodeWavOutput._make(_result)
  return _result

DecodeWav = tf_export("raw_ops.DecodeWav")(_ops.to_raw_op(decode_wav))
_dispatcher_for_decode_wav = decode_wav._tf_type_based_dispatcher.Dispatch


def decode_wav_eager_fallback(contents: Annotated[Any, _atypes.String], desired_channels: int, desired_samples: int, name, ctx):
  if desired_channels is None:
    desired_channels = -1
  desired_channels = _execute.make_int(desired_channels, "desired_channels")
  if desired_samples is None:
    desired_samples = -1
  desired_samples = _execute.make_int(desired_samples, "desired_samples")
  contents = _ops.convert_to_tensor(contents, _dtypes.string)
  _inputs_flat = [contents]
  _attrs = ("desired_channels", desired_channels, "desired_samples",
  desired_samples)
  _result = _execute.execute(b"DecodeWav", 2, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "DecodeWav", _inputs_flat, _attrs, _result)
  _result = _DecodeWavOutput._make(_result)
  return _result


@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('audio.encode_wav')
def encode_wav(audio: Annotated[Any, _atypes.Float32], sample_rate: Annotated[Any, _atypes.Int32], name=None) -> Annotated[Any, _atypes.String]:
  r"""Encode audio data using the WAV file format.

  This operation will generate a string suitable to be saved out to create a .wav
  audio file. It will be encoded in the 16-bit PCM format. It takes in float
  values in the range -1.0f to 1.0f, and any outside that value will be clamped to
  that range.

  `audio` is a 2-D float Tensor of shape `[length, channels]`.
  `sample_rate` is a scalar Tensor holding the rate to use (e.g. 44100).

  Args:
    audio: A `Tensor` of type `float32`. 2-D with shape `[length, channels]`.
    sample_rate: A `Tensor` of type `int32`.
      Scalar containing the sample frequency.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `string`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "EncodeWav", name, audio, sample_rate)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_encode_wav(
          (audio, sample_rate, name,), None)
      if _result is not NotImplemented:
        return _result
      return encode_wav_eager_fallback(
          audio, sample_rate, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            encode_wav, (), dict(audio=audio, sample_rate=sample_rate,
                                 name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_encode_wav(
        (audio, sample_rate, name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "EncodeWav", audio=audio, sample_rate=sample_rate, name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          encode_wav, (), dict(audio=audio, sample_rate=sample_rate,
                               name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ()
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "EncodeWav", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

EncodeWav = tf_export("raw_ops.EncodeWav")(_ops.to_raw_op(encode_wav))
_dispatcher_for_encode_wav = encode_wav._tf_type_based_dispatcher.Dispatch


def encode_wav_eager_fallback(audio: Annotated[Any, _atypes.Float32], sample_rate: Annotated[Any, _atypes.Int32], name, ctx) -> Annotated[Any, _atypes.String]:
  audio = _ops.convert_to_tensor(audio, _dtypes.float32)
  sample_rate = _ops.convert_to_tensor(sample_rate, _dtypes.int32)
  _inputs_flat = [audio, sample_rate]
  _attrs = None
  _result = _execute.execute(b"EncodeWav", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "EncodeWav", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


def mfcc(spectrogram: Annotated[Any, _atypes.Float32], sample_rate: Annotated[Any, _atypes.Int32], upper_frequency_limit:float=4000, lower_frequency_limit:float=20, filterbank_channel_count:int=40, dct_coefficient_count:int=13, name=None) -> Annotated[Any, _atypes.Float32]:
  r"""Transforms a spectrogram into a form that's useful for speech recognition.

  Mel Frequency Cepstral Coefficients are a way of representing audio data that's
  been effective as an input feature for machine learning. They are created by
  taking the spectrum of a spectrogram (a 'cepstrum'), and discarding some of the
  higher frequencies that are less significant to the human ear. They have a long
  history in the speech recognition world, and https://en.wikipedia.org/wiki/Mel-frequency_cepstrum
  is a good resource to learn more.

  Args:
    spectrogram: A `Tensor` of type `float32`.
      Typically produced by the Spectrogram op, with magnitude_squared
      set to true.
    sample_rate: A `Tensor` of type `int32`.
      How many samples per second the source audio used.
    upper_frequency_limit: An optional `float`. Defaults to `4000`.
      The highest frequency to use when calculating the
      ceptstrum.
    lower_frequency_limit: An optional `float`. Defaults to `20`.
      The lowest frequency to use when calculating the
      ceptstrum.
    filterbank_channel_count: An optional `int`. Defaults to `40`.
      Resolution of the Mel bank used internally.
    dct_coefficient_count: An optional `int`. Defaults to `13`.
      How many output channels to produce per time slice.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `float32`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "Mfcc", name, spectrogram, sample_rate, "upper_frequency_limit",
        upper_frequency_limit, "lower_frequency_limit", lower_frequency_limit,
        "filterbank_channel_count", filterbank_channel_count,
        "dct_coefficient_count", dct_coefficient_count)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return mfcc_eager_fallback(
          spectrogram, sample_rate,
          upper_frequency_limit=upper_frequency_limit,
          lower_frequency_limit=lower_frequency_limit,
          filterbank_channel_count=filterbank_channel_count,
          dct_coefficient_count=dct_coefficient_count, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if upper_frequency_limit is None:
    upper_frequency_limit = 4000
  upper_frequency_limit = _execute.make_float(upper_frequency_limit, "upper_frequency_limit")
  if lower_frequency_limit is None:
    lower_frequency_limit = 20
  lower_frequency_limit = _execute.make_float(lower_frequency_limit, "lower_frequency_limit")
  if filterbank_channel_count is None:
    filterbank_channel_count = 40
  filterbank_channel_count = _execute.make_int(filterbank_channel_count, "filterbank_channel_count")
  if dct_coefficient_count is None:
    dct_coefficient_count = 13
  dct_coefficient_count = _execute.make_int(dct_coefficient_count, "dct_coefficient_count")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "Mfcc", spectrogram=spectrogram, sample_rate=sample_rate,
                upper_frequency_limit=upper_frequency_limit,
                lower_frequency_limit=lower_frequency_limit,
                filterbank_channel_count=filterbank_channel_count,
                dct_coefficient_count=dct_coefficient_count, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("upper_frequency_limit", _op.get_attr("upper_frequency_limit"),
              "lower_frequency_limit", _op.get_attr("lower_frequency_limit"),
              "filterbank_channel_count",
              _op._get_attr_int("filterbank_channel_count"),
              "dct_coefficient_count",
              _op._get_attr_int("dct_coefficient_count"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "Mfcc", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

Mfcc = tf_export("raw_ops.Mfcc")(_ops.to_raw_op(mfcc))


def mfcc_eager_fallback(spectrogram: Annotated[Any, _atypes.Float32], sample_rate: Annotated[Any, _atypes.Int32], upper_frequency_limit: float, lower_frequency_limit: float, filterbank_channel_count: int, dct_coefficient_count: int, name, ctx) -> Annotated[Any, _atypes.Float32]:
  if upper_frequency_limit is None:
    upper_frequency_limit = 4000
  upper_frequency_limit = _execute.make_float(upper_frequency_limit, "upper_frequency_limit")
  if lower_frequency_limit is None:
    lower_frequency_limit = 20
  lower_frequency_limit = _execute.make_float(lower_frequency_limit, "lower_frequency_limit")
  if filterbank_channel_count is None:
    filterbank_channel_count = 40
  filterbank_channel_count = _execute.make_int(filterbank_channel_count, "filterbank_channel_count")
  if dct_coefficient_count is None:
    dct_coefficient_count = 13
  dct_coefficient_count = _execute.make_int(dct_coefficient_count, "dct_coefficient_count")
  spectrogram = _ops.convert_to_tensor(spectrogram, _dtypes.float32)
  sample_rate = _ops.convert_to_tensor(sample_rate, _dtypes.int32)
  _inputs_flat = [spectrogram, sample_rate]
  _attrs = ("upper_frequency_limit", upper_frequency_limit,
  "lower_frequency_limit", lower_frequency_limit, "filterbank_channel_count",
  filterbank_channel_count, "dct_coefficient_count", dct_coefficient_count)
  _result = _execute.execute(b"Mfcc", 1, inputs=_inputs_flat, attrs=_attrs,
                             ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "Mfcc", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

