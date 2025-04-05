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

TV_AudioMicrofrontend_out_type = TypeVar("TV_AudioMicrofrontend_out_type", _atypes.Float32, _atypes.UInt16)

@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('audio_microfrontend')
def audio_microfrontend(audio: Annotated[Any, _atypes.Int16], sample_rate:int=16000, window_size:int=25, window_step:int=10, num_channels:int=32, upper_band_limit:float=7500, lower_band_limit:float=125, smoothing_bits:int=10, even_smoothing:float=0.025, odd_smoothing:float=0.06, min_signal_remaining:float=0.05, enable_pcan:bool=False, pcan_strength:float=0.95, pcan_offset:float=80, gain_bits:int=21, enable_log:bool=True, scale_shift:int=6, left_context:int=0, right_context:int=0, frame_stride:int=1, zero_padding:bool=False, out_scale:int=1, out_type:TV_AudioMicrofrontend_out_type=_dtypes.uint16, name=None) -> Annotated[Any, TV_AudioMicrofrontend_out_type]:
  r"""Audio Microfrontend Op.

  This Op converts a sequence of audio data into one or more
  feature vectors containing filterbanks of the input. The
  conversion process uses a lightweight library to perform:

  1. A slicing window function
  2. Short-time FFTs
  3. Filterbank calculations
  4. Noise reduction
  5. PCAN Auto Gain Control
  6. Logarithmic scaling

  Arguments
    audio: 1D Tensor, int16 audio data in temporal ordering.
    sample_rate: Integer, the sample rate of the audio in Hz.
    window_size: Integer, length of desired time frames in ms.
    window_step: Integer, length of step size for the next frame in ms.
    num_channels: Integer, the number of filterbank channels to use.
    upper_band_limit: Float, the highest frequency included in the filterbanks.
    lower_band_limit: Float, the lowest frequency included in the filterbanks.
    smoothing_bits: Int, scale up signal by 2^(smoothing_bits) before reduction.
    even_smoothing: Float, smoothing coefficient for even-numbered channels.
    odd_smoothing: Float, smoothing coefficient for odd-numbered channels.
    min_signal_remaining: Float, fraction of signal to preserve in smoothing.
    enable_pcan: Bool, enable PCAN auto gain control.
    pcan_strength: Float, gain normalization exponent.
    pcan_offset: Float, positive value added in the normalization denominator.
    gain_bits: Int, number of fractional bits in the gain.
    enable_log: Bool, enable logarithmic scaling of filterbanks.
    scale_shift: Integer, scale filterbanks by 2^(scale_shift).
    left_context: Integer, number of preceding frames to attach to each frame.
    right_context: Integer, number of preceding frames to attach to each frame.
    frame_stride: Integer, M frames to skip over, where output[n] = frame[n*M].
    zero_padding: Bool, if left/right context is out-of-bounds, attach frame of
                  zeroes. Otherwise, frame[0] or frame[size-1] will be copied.
    out_scale: Integer, divide all filterbanks by this number.
    out_type: DType, type of the output Tensor, defaults to UINT16.

  Returns
    filterbanks: 2D Tensor, each row is a time frame, each column is a channel.

  Args:
    audio: A `Tensor` of type `int16`.
    sample_rate: An optional `int`. Defaults to `16000`.
    window_size: An optional `int`. Defaults to `25`.
    window_step: An optional `int`. Defaults to `10`.
    num_channels: An optional `int`. Defaults to `32`.
    upper_band_limit: An optional `float`. Defaults to `7500`.
    lower_band_limit: An optional `float`. Defaults to `125`.
    smoothing_bits: An optional `int`. Defaults to `10`.
    even_smoothing: An optional `float`. Defaults to `0.025`.
    odd_smoothing: An optional `float`. Defaults to `0.06`.
    min_signal_remaining: An optional `float`. Defaults to `0.05`.
    enable_pcan: An optional `bool`. Defaults to `False`.
    pcan_strength: An optional `float`. Defaults to `0.95`.
    pcan_offset: An optional `float`. Defaults to `80`.
    gain_bits: An optional `int`. Defaults to `21`.
    enable_log: An optional `bool`. Defaults to `True`.
    scale_shift: An optional `int`. Defaults to `6`.
    left_context: An optional `int`. Defaults to `0`.
    right_context: An optional `int`. Defaults to `0`.
    frame_stride: An optional `int`. Defaults to `1`.
    zero_padding: An optional `bool`. Defaults to `False`.
    out_scale: An optional `int`. Defaults to `1`.
    out_type: An optional `tf.DType` from: `tf.uint16, tf.float32`. Defaults to `tf.uint16`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `out_type`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "AudioMicrofrontend", name, audio, "sample_rate", sample_rate,
        "window_size", window_size, "window_step", window_step,
        "num_channels", num_channels, "upper_band_limit", upper_band_limit,
        "lower_band_limit", lower_band_limit, "smoothing_bits",
        smoothing_bits, "even_smoothing", even_smoothing, "odd_smoothing",
        odd_smoothing, "min_signal_remaining", min_signal_remaining,
        "enable_pcan", enable_pcan, "pcan_strength", pcan_strength,
        "pcan_offset", pcan_offset, "gain_bits", gain_bits, "enable_log",
        enable_log, "scale_shift", scale_shift, "left_context", left_context,
        "right_context", right_context, "frame_stride", frame_stride,
        "zero_padding", zero_padding, "out_scale", out_scale, "out_type",
        out_type)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_audio_microfrontend(
          (audio, sample_rate, window_size, window_step, num_channels,
          upper_band_limit, lower_band_limit, smoothing_bits, even_smoothing,
          odd_smoothing, min_signal_remaining, enable_pcan, pcan_strength,
          pcan_offset, gain_bits, enable_log, scale_shift, left_context,
          right_context, frame_stride, zero_padding, out_scale, out_type,
          name,), None)
      if _result is not NotImplemented:
        return _result
      return audio_microfrontend_eager_fallback(
          audio, sample_rate=sample_rate, window_size=window_size,
          window_step=window_step, num_channels=num_channels,
          upper_band_limit=upper_band_limit,
          lower_band_limit=lower_band_limit, smoothing_bits=smoothing_bits,
          even_smoothing=even_smoothing, odd_smoothing=odd_smoothing,
          min_signal_remaining=min_signal_remaining, enable_pcan=enable_pcan,
          pcan_strength=pcan_strength, pcan_offset=pcan_offset,
          gain_bits=gain_bits, enable_log=enable_log, scale_shift=scale_shift,
          left_context=left_context, right_context=right_context,
          frame_stride=frame_stride, zero_padding=zero_padding,
          out_scale=out_scale, out_type=out_type, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            audio_microfrontend, (), dict(audio=audio,
                                          sample_rate=sample_rate,
                                          window_size=window_size,
                                          window_step=window_step,
                                          num_channels=num_channels,
                                          upper_band_limit=upper_band_limit,
                                          lower_band_limit=lower_band_limit,
                                          smoothing_bits=smoothing_bits,
                                          even_smoothing=even_smoothing,
                                          odd_smoothing=odd_smoothing,
                                          min_signal_remaining=min_signal_remaining,
                                          enable_pcan=enable_pcan,
                                          pcan_strength=pcan_strength,
                                          pcan_offset=pcan_offset,
                                          gain_bits=gain_bits,
                                          enable_log=enable_log,
                                          scale_shift=scale_shift,
                                          left_context=left_context,
                                          right_context=right_context,
                                          frame_stride=frame_stride,
                                          zero_padding=zero_padding,
                                          out_scale=out_scale,
                                          out_type=out_type, name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_audio_microfrontend(
        (audio, sample_rate, window_size, window_step, num_channels,
        upper_band_limit, lower_band_limit, smoothing_bits, even_smoothing,
        odd_smoothing, min_signal_remaining, enable_pcan, pcan_strength,
        pcan_offset, gain_bits, enable_log, scale_shift, left_context,
        right_context, frame_stride, zero_padding, out_scale, out_type,
        name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  if sample_rate is None:
    sample_rate = 16000
  sample_rate = _execute.make_int(sample_rate, "sample_rate")
  if window_size is None:
    window_size = 25
  window_size = _execute.make_int(window_size, "window_size")
  if window_step is None:
    window_step = 10
  window_step = _execute.make_int(window_step, "window_step")
  if num_channels is None:
    num_channels = 32
  num_channels = _execute.make_int(num_channels, "num_channels")
  if upper_band_limit is None:
    upper_band_limit = 7500
  upper_band_limit = _execute.make_float(upper_band_limit, "upper_band_limit")
  if lower_band_limit is None:
    lower_band_limit = 125
  lower_band_limit = _execute.make_float(lower_band_limit, "lower_band_limit")
  if smoothing_bits is None:
    smoothing_bits = 10
  smoothing_bits = _execute.make_int(smoothing_bits, "smoothing_bits")
  if even_smoothing is None:
    even_smoothing = 0.025
  even_smoothing = _execute.make_float(even_smoothing, "even_smoothing")
  if odd_smoothing is None:
    odd_smoothing = 0.06
  odd_smoothing = _execute.make_float(odd_smoothing, "odd_smoothing")
  if min_signal_remaining is None:
    min_signal_remaining = 0.05
  min_signal_remaining = _execute.make_float(min_signal_remaining, "min_signal_remaining")
  if enable_pcan is None:
    enable_pcan = False
  enable_pcan = _execute.make_bool(enable_pcan, "enable_pcan")
  if pcan_strength is None:
    pcan_strength = 0.95
  pcan_strength = _execute.make_float(pcan_strength, "pcan_strength")
  if pcan_offset is None:
    pcan_offset = 80
  pcan_offset = _execute.make_float(pcan_offset, "pcan_offset")
  if gain_bits is None:
    gain_bits = 21
  gain_bits = _execute.make_int(gain_bits, "gain_bits")
  if enable_log is None:
    enable_log = True
  enable_log = _execute.make_bool(enable_log, "enable_log")
  if scale_shift is None:
    scale_shift = 6
  scale_shift = _execute.make_int(scale_shift, "scale_shift")
  if left_context is None:
    left_context = 0
  left_context = _execute.make_int(left_context, "left_context")
  if right_context is None:
    right_context = 0
  right_context = _execute.make_int(right_context, "right_context")
  if frame_stride is None:
    frame_stride = 1
  frame_stride = _execute.make_int(frame_stride, "frame_stride")
  if zero_padding is None:
    zero_padding = False
  zero_padding = _execute.make_bool(zero_padding, "zero_padding")
  if out_scale is None:
    out_scale = 1
  out_scale = _execute.make_int(out_scale, "out_scale")
  if out_type is None:
    out_type = _dtypes.uint16
  out_type = _execute.make_type(out_type, "out_type")
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "AudioMicrofrontend", audio=audio, sample_rate=sample_rate,
                              window_size=window_size,
                              window_step=window_step,
                              num_channels=num_channels,
                              upper_band_limit=upper_band_limit,
                              lower_band_limit=lower_band_limit,
                              smoothing_bits=smoothing_bits,
                              even_smoothing=even_smoothing,
                              odd_smoothing=odd_smoothing,
                              min_signal_remaining=min_signal_remaining,
                              enable_pcan=enable_pcan,
                              pcan_strength=pcan_strength,
                              pcan_offset=pcan_offset, gain_bits=gain_bits,
                              enable_log=enable_log, scale_shift=scale_shift,
                              left_context=left_context,
                              right_context=right_context,
                              frame_stride=frame_stride,
                              zero_padding=zero_padding, out_scale=out_scale,
                              out_type=out_type, name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          audio_microfrontend, (), dict(audio=audio, sample_rate=sample_rate,
                                        window_size=window_size,
                                        window_step=window_step,
                                        num_channels=num_channels,
                                        upper_band_limit=upper_band_limit,
                                        lower_band_limit=lower_band_limit,
                                        smoothing_bits=smoothing_bits,
                                        even_smoothing=even_smoothing,
                                        odd_smoothing=odd_smoothing,
                                        min_signal_remaining=min_signal_remaining,
                                        enable_pcan=enable_pcan,
                                        pcan_strength=pcan_strength,
                                        pcan_offset=pcan_offset,
                                        gain_bits=gain_bits,
                                        enable_log=enable_log,
                                        scale_shift=scale_shift,
                                        left_context=left_context,
                                        right_context=right_context,
                                        frame_stride=frame_stride,
                                        zero_padding=zero_padding,
                                        out_scale=out_scale,
                                        out_type=out_type, name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("sample_rate", _op._get_attr_int("sample_rate"), "window_size",
              _op._get_attr_int("window_size"), "window_step",
              _op._get_attr_int("window_step"), "num_channels",
              _op._get_attr_int("num_channels"), "upper_band_limit",
              _op.get_attr("upper_band_limit"), "lower_band_limit",
              _op.get_attr("lower_band_limit"), "smoothing_bits",
              _op._get_attr_int("smoothing_bits"), "even_smoothing",
              _op.get_attr("even_smoothing"), "odd_smoothing",
              _op.get_attr("odd_smoothing"), "min_signal_remaining",
              _op.get_attr("min_signal_remaining"), "enable_pcan",
              _op._get_attr_bool("enable_pcan"), "pcan_strength",
              _op.get_attr("pcan_strength"), "pcan_offset",
              _op.get_attr("pcan_offset"), "gain_bits",
              _op._get_attr_int("gain_bits"), "enable_log",
              _op._get_attr_bool("enable_log"), "scale_shift",
              _op._get_attr_int("scale_shift"), "left_context",
              _op._get_attr_int("left_context"), "right_context",
              _op._get_attr_int("right_context"), "frame_stride",
              _op._get_attr_int("frame_stride"), "zero_padding",
              _op._get_attr_bool("zero_padding"), "out_scale",
              _op._get_attr_int("out_scale"), "out_type",
              _op._get_attr_type("out_type"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "AudioMicrofrontend", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

AudioMicrofrontend = tf_export("raw_ops.AudioMicrofrontend")(_ops.to_raw_op(audio_microfrontend))
_dispatcher_for_audio_microfrontend = audio_microfrontend._tf_type_based_dispatcher.Dispatch


def audio_microfrontend_eager_fallback(audio: Annotated[Any, _atypes.Int16], sample_rate: int, window_size: int, window_step: int, num_channels: int, upper_band_limit: float, lower_band_limit: float, smoothing_bits: int, even_smoothing: float, odd_smoothing: float, min_signal_remaining: float, enable_pcan: bool, pcan_strength: float, pcan_offset: float, gain_bits: int, enable_log: bool, scale_shift: int, left_context: int, right_context: int, frame_stride: int, zero_padding: bool, out_scale: int, out_type: TV_AudioMicrofrontend_out_type, name, ctx) -> Annotated[Any, TV_AudioMicrofrontend_out_type]:
  if sample_rate is None:
    sample_rate = 16000
  sample_rate = _execute.make_int(sample_rate, "sample_rate")
  if window_size is None:
    window_size = 25
  window_size = _execute.make_int(window_size, "window_size")
  if window_step is None:
    window_step = 10
  window_step = _execute.make_int(window_step, "window_step")
  if num_channels is None:
    num_channels = 32
  num_channels = _execute.make_int(num_channels, "num_channels")
  if upper_band_limit is None:
    upper_band_limit = 7500
  upper_band_limit = _execute.make_float(upper_band_limit, "upper_band_limit")
  if lower_band_limit is None:
    lower_band_limit = 125
  lower_band_limit = _execute.make_float(lower_band_limit, "lower_band_limit")
  if smoothing_bits is None:
    smoothing_bits = 10
  smoothing_bits = _execute.make_int(smoothing_bits, "smoothing_bits")
  if even_smoothing is None:
    even_smoothing = 0.025
  even_smoothing = _execute.make_float(even_smoothing, "even_smoothing")
  if odd_smoothing is None:
    odd_smoothing = 0.06
  odd_smoothing = _execute.make_float(odd_smoothing, "odd_smoothing")
  if min_signal_remaining is None:
    min_signal_remaining = 0.05
  min_signal_remaining = _execute.make_float(min_signal_remaining, "min_signal_remaining")
  if enable_pcan is None:
    enable_pcan = False
  enable_pcan = _execute.make_bool(enable_pcan, "enable_pcan")
  if pcan_strength is None:
    pcan_strength = 0.95
  pcan_strength = _execute.make_float(pcan_strength, "pcan_strength")
  if pcan_offset is None:
    pcan_offset = 80
  pcan_offset = _execute.make_float(pcan_offset, "pcan_offset")
  if gain_bits is None:
    gain_bits = 21
  gain_bits = _execute.make_int(gain_bits, "gain_bits")
  if enable_log is None:
    enable_log = True
  enable_log = _execute.make_bool(enable_log, "enable_log")
  if scale_shift is None:
    scale_shift = 6
  scale_shift = _execute.make_int(scale_shift, "scale_shift")
  if left_context is None:
    left_context = 0
  left_context = _execute.make_int(left_context, "left_context")
  if right_context is None:
    right_context = 0
  right_context = _execute.make_int(right_context, "right_context")
  if frame_stride is None:
    frame_stride = 1
  frame_stride = _execute.make_int(frame_stride, "frame_stride")
  if zero_padding is None:
    zero_padding = False
  zero_padding = _execute.make_bool(zero_padding, "zero_padding")
  if out_scale is None:
    out_scale = 1
  out_scale = _execute.make_int(out_scale, "out_scale")
  if out_type is None:
    out_type = _dtypes.uint16
  out_type = _execute.make_type(out_type, "out_type")
  audio = _ops.convert_to_tensor(audio, _dtypes.int16)
  _inputs_flat = [audio]
  _attrs = ("sample_rate", sample_rate, "window_size", window_size,
  "window_step", window_step, "num_channels", num_channels,
  "upper_band_limit", upper_band_limit, "lower_band_limit", lower_band_limit,
  "smoothing_bits", smoothing_bits, "even_smoothing", even_smoothing,
  "odd_smoothing", odd_smoothing, "min_signal_remaining",
  min_signal_remaining, "enable_pcan", enable_pcan, "pcan_strength",
  pcan_strength, "pcan_offset", pcan_offset, "gain_bits", gain_bits,
  "enable_log", enable_log, "scale_shift", scale_shift, "left_context",
  left_context, "right_context", right_context, "frame_stride", frame_stride,
  "zero_padding", zero_padding, "out_scale", out_scale, "out_type", out_type)
  _result = _execute.execute(b"AudioMicrofrontend", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "AudioMicrofrontend", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

