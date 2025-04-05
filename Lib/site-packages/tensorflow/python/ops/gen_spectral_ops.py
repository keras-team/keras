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

def batch_fft(input: Annotated[Any, _atypes.Complex64], name=None) -> Annotated[Any, _atypes.Complex64]:
  r"""TODO: add doc.

  Args:
    input: A `Tensor` of type `complex64`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `complex64`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "BatchFFT", name, input)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return batch_fft_eager_fallback(
          input, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "BatchFFT", input=input, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ()
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "BatchFFT", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

BatchFFT = tf_export("raw_ops.BatchFFT")(_ops.to_raw_op(batch_fft))


def batch_fft_eager_fallback(input: Annotated[Any, _atypes.Complex64], name, ctx) -> Annotated[Any, _atypes.Complex64]:
  input = _ops.convert_to_tensor(input, _dtypes.complex64)
  _inputs_flat = [input]
  _attrs = None
  _result = _execute.execute(b"BatchFFT", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "BatchFFT", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


def batch_fft2d(input: Annotated[Any, _atypes.Complex64], name=None) -> Annotated[Any, _atypes.Complex64]:
  r"""TODO: add doc.

  Args:
    input: A `Tensor` of type `complex64`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `complex64`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "BatchFFT2D", name, input)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return batch_fft2d_eager_fallback(
          input, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "BatchFFT2D", input=input, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ()
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "BatchFFT2D", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

BatchFFT2D = tf_export("raw_ops.BatchFFT2D")(_ops.to_raw_op(batch_fft2d))


def batch_fft2d_eager_fallback(input: Annotated[Any, _atypes.Complex64], name, ctx) -> Annotated[Any, _atypes.Complex64]:
  input = _ops.convert_to_tensor(input, _dtypes.complex64)
  _inputs_flat = [input]
  _attrs = None
  _result = _execute.execute(b"BatchFFT2D", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "BatchFFT2D", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


def batch_fft3d(input: Annotated[Any, _atypes.Complex64], name=None) -> Annotated[Any, _atypes.Complex64]:
  r"""TODO: add doc.

  Args:
    input: A `Tensor` of type `complex64`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `complex64`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "BatchFFT3D", name, input)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return batch_fft3d_eager_fallback(
          input, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "BatchFFT3D", input=input, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ()
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "BatchFFT3D", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

BatchFFT3D = tf_export("raw_ops.BatchFFT3D")(_ops.to_raw_op(batch_fft3d))


def batch_fft3d_eager_fallback(input: Annotated[Any, _atypes.Complex64], name, ctx) -> Annotated[Any, _atypes.Complex64]:
  input = _ops.convert_to_tensor(input, _dtypes.complex64)
  _inputs_flat = [input]
  _attrs = None
  _result = _execute.execute(b"BatchFFT3D", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "BatchFFT3D", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


def batch_ifft(input: Annotated[Any, _atypes.Complex64], name=None) -> Annotated[Any, _atypes.Complex64]:
  r"""TODO: add doc.

  Args:
    input: A `Tensor` of type `complex64`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `complex64`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "BatchIFFT", name, input)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return batch_ifft_eager_fallback(
          input, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "BatchIFFT", input=input, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ()
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "BatchIFFT", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

BatchIFFT = tf_export("raw_ops.BatchIFFT")(_ops.to_raw_op(batch_ifft))


def batch_ifft_eager_fallback(input: Annotated[Any, _atypes.Complex64], name, ctx) -> Annotated[Any, _atypes.Complex64]:
  input = _ops.convert_to_tensor(input, _dtypes.complex64)
  _inputs_flat = [input]
  _attrs = None
  _result = _execute.execute(b"BatchIFFT", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "BatchIFFT", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


def batch_ifft2d(input: Annotated[Any, _atypes.Complex64], name=None) -> Annotated[Any, _atypes.Complex64]:
  r"""TODO: add doc.

  Args:
    input: A `Tensor` of type `complex64`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `complex64`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "BatchIFFT2D", name, input)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return batch_ifft2d_eager_fallback(
          input, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "BatchIFFT2D", input=input, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ()
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "BatchIFFT2D", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

BatchIFFT2D = tf_export("raw_ops.BatchIFFT2D")(_ops.to_raw_op(batch_ifft2d))


def batch_ifft2d_eager_fallback(input: Annotated[Any, _atypes.Complex64], name, ctx) -> Annotated[Any, _atypes.Complex64]:
  input = _ops.convert_to_tensor(input, _dtypes.complex64)
  _inputs_flat = [input]
  _attrs = None
  _result = _execute.execute(b"BatchIFFT2D", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "BatchIFFT2D", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


def batch_ifft3d(input: Annotated[Any, _atypes.Complex64], name=None) -> Annotated[Any, _atypes.Complex64]:
  r"""TODO: add doc.

  Args:
    input: A `Tensor` of type `complex64`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `complex64`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "BatchIFFT3D", name, input)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return batch_ifft3d_eager_fallback(
          input, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "BatchIFFT3D", input=input, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ()
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "BatchIFFT3D", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

BatchIFFT3D = tf_export("raw_ops.BatchIFFT3D")(_ops.to_raw_op(batch_ifft3d))


def batch_ifft3d_eager_fallback(input: Annotated[Any, _atypes.Complex64], name, ctx) -> Annotated[Any, _atypes.Complex64]:
  input = _ops.convert_to_tensor(input, _dtypes.complex64)
  _inputs_flat = [input]
  _attrs = None
  _result = _execute.execute(b"BatchIFFT3D", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "BatchIFFT3D", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_FFT_Tcomplex = TypeVar("TV_FFT_Tcomplex", _atypes.Complex128, _atypes.Complex64)

@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('signal.fft', v1=['signal.fft', 'spectral.fft', 'fft'])
@deprecated_endpoints('spectral.fft', 'fft')
def fft(input: Annotated[Any, TV_FFT_Tcomplex], name=None) -> Annotated[Any, TV_FFT_Tcomplex]:
  r"""Fast Fourier transform.

  Computes the 1-dimensional discrete Fourier transform over the inner-most
  dimension of `input`.

  Args:
    input: A `Tensor`. Must be one of the following types: `complex64`, `complex128`.
      A complex tensor.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "FFT", name, input)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_fft(
          (input, name,), None)
      if _result is not NotImplemented:
        return _result
      return fft_eager_fallback(
          input, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            fft, (), dict(input=input, name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_fft(
        (input, name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "FFT", input=input, name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          fft, (), dict(input=input, name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("Tcomplex", _op._get_attr_type("Tcomplex"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "FFT", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

FFT = tf_export("raw_ops.FFT")(_ops.to_raw_op(fft))
_dispatcher_for_fft = fft._tf_type_based_dispatcher.Dispatch


def fft_eager_fallback(input: Annotated[Any, TV_FFT_Tcomplex], name, ctx) -> Annotated[Any, TV_FFT_Tcomplex]:
  _attr_Tcomplex, (input,) = _execute.args_to_matching_eager([input], ctx, [_dtypes.complex64, _dtypes.complex128, ], _dtypes.complex64)
  _inputs_flat = [input]
  _attrs = ("Tcomplex", _attr_Tcomplex)
  _result = _execute.execute(b"FFT", 1, inputs=_inputs_flat, attrs=_attrs,
                             ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "FFT", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_FFT2D_Tcomplex = TypeVar("TV_FFT2D_Tcomplex", _atypes.Complex128, _atypes.Complex64)

@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('signal.fft2d', v1=['signal.fft2d', 'spectral.fft2d', 'fft2d'])
@deprecated_endpoints('spectral.fft2d', 'fft2d')
def fft2d(input: Annotated[Any, TV_FFT2D_Tcomplex], name=None) -> Annotated[Any, TV_FFT2D_Tcomplex]:
  r"""2D fast Fourier transform.

  Computes the 2-dimensional discrete Fourier transform over the inner-most
  2 dimensions of `input`.

  Args:
    input: A `Tensor`. Must be one of the following types: `complex64`, `complex128`.
      A complex tensor.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "FFT2D", name, input)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_fft2d(
          (input, name,), None)
      if _result is not NotImplemented:
        return _result
      return fft2d_eager_fallback(
          input, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            fft2d, (), dict(input=input, name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_fft2d(
        (input, name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "FFT2D", input=input, name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          fft2d, (), dict(input=input, name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("Tcomplex", _op._get_attr_type("Tcomplex"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "FFT2D", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

FFT2D = tf_export("raw_ops.FFT2D")(_ops.to_raw_op(fft2d))
_dispatcher_for_fft2d = fft2d._tf_type_based_dispatcher.Dispatch


def fft2d_eager_fallback(input: Annotated[Any, TV_FFT2D_Tcomplex], name, ctx) -> Annotated[Any, TV_FFT2D_Tcomplex]:
  _attr_Tcomplex, (input,) = _execute.args_to_matching_eager([input], ctx, [_dtypes.complex64, _dtypes.complex128, ], _dtypes.complex64)
  _inputs_flat = [input]
  _attrs = ("Tcomplex", _attr_Tcomplex)
  _result = _execute.execute(b"FFT2D", 1, inputs=_inputs_flat, attrs=_attrs,
                             ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "FFT2D", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_FFT3D_Tcomplex = TypeVar("TV_FFT3D_Tcomplex", _atypes.Complex128, _atypes.Complex64)

@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('signal.fft3d', v1=['signal.fft3d', 'spectral.fft3d', 'fft3d'])
@deprecated_endpoints('spectral.fft3d', 'fft3d')
def fft3d(input: Annotated[Any, TV_FFT3D_Tcomplex], name=None) -> Annotated[Any, TV_FFT3D_Tcomplex]:
  r"""3D fast Fourier transform.

  Computes the 3-dimensional discrete Fourier transform over the inner-most 3
  dimensions of `input`.

  Args:
    input: A `Tensor`. Must be one of the following types: `complex64`, `complex128`.
      A complex tensor.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "FFT3D", name, input)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_fft3d(
          (input, name,), None)
      if _result is not NotImplemented:
        return _result
      return fft3d_eager_fallback(
          input, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            fft3d, (), dict(input=input, name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_fft3d(
        (input, name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "FFT3D", input=input, name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          fft3d, (), dict(input=input, name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("Tcomplex", _op._get_attr_type("Tcomplex"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "FFT3D", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

FFT3D = tf_export("raw_ops.FFT3D")(_ops.to_raw_op(fft3d))
_dispatcher_for_fft3d = fft3d._tf_type_based_dispatcher.Dispatch


def fft3d_eager_fallback(input: Annotated[Any, TV_FFT3D_Tcomplex], name, ctx) -> Annotated[Any, TV_FFT3D_Tcomplex]:
  _attr_Tcomplex, (input,) = _execute.args_to_matching_eager([input], ctx, [_dtypes.complex64, _dtypes.complex128, ], _dtypes.complex64)
  _inputs_flat = [input]
  _attrs = ("Tcomplex", _attr_Tcomplex)
  _result = _execute.execute(b"FFT3D", 1, inputs=_inputs_flat, attrs=_attrs,
                             ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "FFT3D", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_FFTND_Tcomplex = TypeVar("TV_FFTND_Tcomplex", _atypes.Complex128, _atypes.Complex64)

@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('fftnd')
def fftnd(input: Annotated[Any, TV_FFTND_Tcomplex], fft_length: Annotated[Any, _atypes.Int32], axes: Annotated[Any, _atypes.Int32], name=None) -> Annotated[Any, TV_FFTND_Tcomplex]:
  r"""ND fast Fourier transform.

  Computes the n-dimensional discrete Fourier transform over
  designated dimensions of `input`. The designated dimensions of
  `input` are assumed to be the result of `FFTND`.

  If fft_length[i]<shape(input)[i], the input is cropped. If
  fft_length[i]>shape(input)[i], the input is padded with zeros. If fft_length
  is not given, the default shape(input) is used.

  Axes mean the dimensions to perform the transform on. Default is to perform on
  all axes.

  Args:
    input: A `Tensor`. Must be one of the following types: `complex64`, `complex128`.
      A complex tensor.
    fft_length: A `Tensor` of type `int32`.
      An int32 tensor. The FFT length for each dimension.
    axes: A `Tensor` of type `int32`.
      An int32 tensor with a same shape as fft_length. Axes to perform the transform.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "FFTND", name, input, fft_length, axes)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_fftnd(
          (input, fft_length, axes, name,), None)
      if _result is not NotImplemented:
        return _result
      return fftnd_eager_fallback(
          input, fft_length, axes, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            fftnd, (), dict(input=input, fft_length=fft_length, axes=axes,
                            name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_fftnd(
        (input, fft_length, axes, name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "FFTND", input=input, fft_length=fft_length, axes=axes, name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          fftnd, (), dict(input=input, fft_length=fft_length, axes=axes,
                          name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("Tcomplex", _op._get_attr_type("Tcomplex"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "FFTND", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

FFTND = tf_export("raw_ops.FFTND")(_ops.to_raw_op(fftnd))
_dispatcher_for_fftnd = fftnd._tf_type_based_dispatcher.Dispatch


def fftnd_eager_fallback(input: Annotated[Any, TV_FFTND_Tcomplex], fft_length: Annotated[Any, _atypes.Int32], axes: Annotated[Any, _atypes.Int32], name, ctx) -> Annotated[Any, TV_FFTND_Tcomplex]:
  _attr_Tcomplex, (input,) = _execute.args_to_matching_eager([input], ctx, [_dtypes.complex64, _dtypes.complex128, ], _dtypes.complex64)
  fft_length = _ops.convert_to_tensor(fft_length, _dtypes.int32)
  axes = _ops.convert_to_tensor(axes, _dtypes.int32)
  _inputs_flat = [input, fft_length, axes]
  _attrs = ("Tcomplex", _attr_Tcomplex)
  _result = _execute.execute(b"FFTND", 1, inputs=_inputs_flat, attrs=_attrs,
                             ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "FFTND", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_IFFT_Tcomplex = TypeVar("TV_IFFT_Tcomplex", _atypes.Complex128, _atypes.Complex64)

@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('signal.ifft', v1=['signal.ifft', 'spectral.ifft', 'ifft'])
@deprecated_endpoints('spectral.ifft', 'ifft')
def ifft(input: Annotated[Any, TV_IFFT_Tcomplex], name=None) -> Annotated[Any, TV_IFFT_Tcomplex]:
  r"""Inverse fast Fourier transform.

  Computes the inverse 1-dimensional discrete Fourier transform over the
  inner-most dimension of `input`.

  Args:
    input: A `Tensor`. Must be one of the following types: `complex64`, `complex128`.
      A complex tensor.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "IFFT", name, input)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_ifft(
          (input, name,), None)
      if _result is not NotImplemented:
        return _result
      return ifft_eager_fallback(
          input, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            ifft, (), dict(input=input, name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_ifft(
        (input, name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "IFFT", input=input, name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          ifft, (), dict(input=input, name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("Tcomplex", _op._get_attr_type("Tcomplex"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "IFFT", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

IFFT = tf_export("raw_ops.IFFT")(_ops.to_raw_op(ifft))
_dispatcher_for_ifft = ifft._tf_type_based_dispatcher.Dispatch


def ifft_eager_fallback(input: Annotated[Any, TV_IFFT_Tcomplex], name, ctx) -> Annotated[Any, TV_IFFT_Tcomplex]:
  _attr_Tcomplex, (input,) = _execute.args_to_matching_eager([input], ctx, [_dtypes.complex64, _dtypes.complex128, ], _dtypes.complex64)
  _inputs_flat = [input]
  _attrs = ("Tcomplex", _attr_Tcomplex)
  _result = _execute.execute(b"IFFT", 1, inputs=_inputs_flat, attrs=_attrs,
                             ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "IFFT", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_IFFT2D_Tcomplex = TypeVar("TV_IFFT2D_Tcomplex", _atypes.Complex128, _atypes.Complex64)

@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('signal.ifft2d', v1=['signal.ifft2d', 'spectral.ifft2d', 'ifft2d'])
@deprecated_endpoints('spectral.ifft2d', 'ifft2d')
def ifft2d(input: Annotated[Any, TV_IFFT2D_Tcomplex], name=None) -> Annotated[Any, TV_IFFT2D_Tcomplex]:
  r"""Inverse 2D fast Fourier transform.

  Computes the inverse 2-dimensional discrete Fourier transform over the
  inner-most 2 dimensions of `input`.

  Args:
    input: A `Tensor`. Must be one of the following types: `complex64`, `complex128`.
      A complex tensor.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "IFFT2D", name, input)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_ifft2d(
          (input, name,), None)
      if _result is not NotImplemented:
        return _result
      return ifft2d_eager_fallback(
          input, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            ifft2d, (), dict(input=input, name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_ifft2d(
        (input, name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "IFFT2D", input=input, name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          ifft2d, (), dict(input=input, name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("Tcomplex", _op._get_attr_type("Tcomplex"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "IFFT2D", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

IFFT2D = tf_export("raw_ops.IFFT2D")(_ops.to_raw_op(ifft2d))
_dispatcher_for_ifft2d = ifft2d._tf_type_based_dispatcher.Dispatch


def ifft2d_eager_fallback(input: Annotated[Any, TV_IFFT2D_Tcomplex], name, ctx) -> Annotated[Any, TV_IFFT2D_Tcomplex]:
  _attr_Tcomplex, (input,) = _execute.args_to_matching_eager([input], ctx, [_dtypes.complex64, _dtypes.complex128, ], _dtypes.complex64)
  _inputs_flat = [input]
  _attrs = ("Tcomplex", _attr_Tcomplex)
  _result = _execute.execute(b"IFFT2D", 1, inputs=_inputs_flat, attrs=_attrs,
                             ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "IFFT2D", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_IFFT3D_Tcomplex = TypeVar("TV_IFFT3D_Tcomplex", _atypes.Complex128, _atypes.Complex64)

@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('signal.ifft3d', v1=['signal.ifft3d', 'spectral.ifft3d', 'ifft3d'])
@deprecated_endpoints('spectral.ifft3d', 'ifft3d')
def ifft3d(input: Annotated[Any, TV_IFFT3D_Tcomplex], name=None) -> Annotated[Any, TV_IFFT3D_Tcomplex]:
  r"""Inverse 3D fast Fourier transform.

  Computes the inverse 3-dimensional discrete Fourier transform over the
  inner-most 3 dimensions of `input`.

  Args:
    input: A `Tensor`. Must be one of the following types: `complex64`, `complex128`.
      A complex tensor.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "IFFT3D", name, input)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_ifft3d(
          (input, name,), None)
      if _result is not NotImplemented:
        return _result
      return ifft3d_eager_fallback(
          input, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            ifft3d, (), dict(input=input, name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_ifft3d(
        (input, name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "IFFT3D", input=input, name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          ifft3d, (), dict(input=input, name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("Tcomplex", _op._get_attr_type("Tcomplex"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "IFFT3D", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

IFFT3D = tf_export("raw_ops.IFFT3D")(_ops.to_raw_op(ifft3d))
_dispatcher_for_ifft3d = ifft3d._tf_type_based_dispatcher.Dispatch


def ifft3d_eager_fallback(input: Annotated[Any, TV_IFFT3D_Tcomplex], name, ctx) -> Annotated[Any, TV_IFFT3D_Tcomplex]:
  _attr_Tcomplex, (input,) = _execute.args_to_matching_eager([input], ctx, [_dtypes.complex64, _dtypes.complex128, ], _dtypes.complex64)
  _inputs_flat = [input]
  _attrs = ("Tcomplex", _attr_Tcomplex)
  _result = _execute.execute(b"IFFT3D", 1, inputs=_inputs_flat, attrs=_attrs,
                             ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "IFFT3D", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_IFFTND_Tcomplex = TypeVar("TV_IFFTND_Tcomplex", _atypes.Complex128, _atypes.Complex64)

@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('ifftnd')
def ifftnd(input: Annotated[Any, TV_IFFTND_Tcomplex], fft_length: Annotated[Any, _atypes.Int32], axes: Annotated[Any, _atypes.Int32], name=None) -> Annotated[Any, TV_IFFTND_Tcomplex]:
  r"""ND inverse fast Fourier transform.

  Computes the n-dimensional inverse discrete Fourier transform over designated
  dimensions of `input`. The designated dimensions of `input` are assumed to be
  the result of `IFFTND`.

  If fft_length[i]<shape(input)[i], the input is cropped. If
  fft_length[i]>shape(input)[i], the input is padded with zeros. If fft_length
  is not given, the default shape(input) is used.

  Axes mean the dimensions to perform the transform on. Default is to perform on
  all axes.

  Args:
    input: A `Tensor`. Must be one of the following types: `complex64`, `complex128`.
      A complex tensor.
    fft_length: A `Tensor` of type `int32`.
      An int32 tensor. The FFT length for each dimension.
    axes: A `Tensor` of type `int32`.
      An int32 tensor with a same shape as fft_length. Axes to perform the transform.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "IFFTND", name, input, fft_length, axes)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_ifftnd(
          (input, fft_length, axes, name,), None)
      if _result is not NotImplemented:
        return _result
      return ifftnd_eager_fallback(
          input, fft_length, axes, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            ifftnd, (), dict(input=input, fft_length=fft_length, axes=axes,
                             name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_ifftnd(
        (input, fft_length, axes, name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "IFFTND", input=input, fft_length=fft_length, axes=axes, name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          ifftnd, (), dict(input=input, fft_length=fft_length, axes=axes,
                           name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("Tcomplex", _op._get_attr_type("Tcomplex"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "IFFTND", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

IFFTND = tf_export("raw_ops.IFFTND")(_ops.to_raw_op(ifftnd))
_dispatcher_for_ifftnd = ifftnd._tf_type_based_dispatcher.Dispatch


def ifftnd_eager_fallback(input: Annotated[Any, TV_IFFTND_Tcomplex], fft_length: Annotated[Any, _atypes.Int32], axes: Annotated[Any, _atypes.Int32], name, ctx) -> Annotated[Any, TV_IFFTND_Tcomplex]:
  _attr_Tcomplex, (input,) = _execute.args_to_matching_eager([input], ctx, [_dtypes.complex64, _dtypes.complex128, ], _dtypes.complex64)
  fft_length = _ops.convert_to_tensor(fft_length, _dtypes.int32)
  axes = _ops.convert_to_tensor(axes, _dtypes.int32)
  _inputs_flat = [input, fft_length, axes]
  _attrs = ("Tcomplex", _attr_Tcomplex)
  _result = _execute.execute(b"IFFTND", 1, inputs=_inputs_flat, attrs=_attrs,
                             ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "IFFTND", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_IRFFT_Treal = TypeVar("TV_IRFFT_Treal", _atypes.Float32, _atypes.Float64)
TV_IRFFT_Tcomplex = TypeVar("TV_IRFFT_Tcomplex", _atypes.Complex128, _atypes.Complex64)

def irfft(input: Annotated[Any, TV_IRFFT_Tcomplex], fft_length: Annotated[Any, _atypes.Int32], Treal:TV_IRFFT_Treal=_dtypes.float32, name=None) -> Annotated[Any, TV_IRFFT_Treal]:
  r"""Inverse real-valued fast Fourier transform.

  Computes the inverse 1-dimensional discrete Fourier transform of a real-valued
  signal over the inner-most dimension of `input`.

  The inner-most dimension of `input` is assumed to be the result of `RFFT`: the
  `fft_length / 2 + 1` unique components of the DFT of a real-valued signal. If
  `fft_length` is not provided, it is computed from the size of the inner-most
  dimension of `input` (`fft_length = 2 * (inner - 1)`). If the FFT length used to
  compute `input` is odd, it should be provided since it cannot be inferred
  properly.

  Along the axis `IRFFT` is computed on, if `fft_length / 2 + 1` is smaller
  than the corresponding dimension of `input`, the dimension is cropped. If it is
  larger, the dimension is padded with zeros.

  Args:
    input: A `Tensor`. Must be one of the following types: `complex64`, `complex128`.
      A complex tensor.
    fft_length: A `Tensor` of type `int32`.
      An int32 tensor of shape [1]. The FFT length.
    Treal: An optional `tf.DType` from: `tf.float32, tf.float64`. Defaults to `tf.float32`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `Treal`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "IRFFT", name, input, fft_length, "Treal", Treal)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return irfft_eager_fallback(
          input, fft_length, Treal=Treal, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if Treal is None:
    Treal = _dtypes.float32
  Treal = _execute.make_type(Treal, "Treal")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "IRFFT", input=input, fft_length=fft_length, Treal=Treal, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("Treal", _op._get_attr_type("Treal"), "Tcomplex",
              _op._get_attr_type("Tcomplex"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "IRFFT", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

IRFFT = tf_export("raw_ops.IRFFT")(_ops.to_raw_op(irfft))


def irfft_eager_fallback(input: Annotated[Any, TV_IRFFT_Tcomplex], fft_length: Annotated[Any, _atypes.Int32], Treal: TV_IRFFT_Treal, name, ctx) -> Annotated[Any, TV_IRFFT_Treal]:
  if Treal is None:
    Treal = _dtypes.float32
  Treal = _execute.make_type(Treal, "Treal")
  _attr_Tcomplex, (input,) = _execute.args_to_matching_eager([input], ctx, [_dtypes.complex64, _dtypes.complex128, ], _dtypes.complex64)
  fft_length = _ops.convert_to_tensor(fft_length, _dtypes.int32)
  _inputs_flat = [input, fft_length]
  _attrs = ("Treal", Treal, "Tcomplex", _attr_Tcomplex)
  _result = _execute.execute(b"IRFFT", 1, inputs=_inputs_flat, attrs=_attrs,
                             ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "IRFFT", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_IRFFT2D_Treal = TypeVar("TV_IRFFT2D_Treal", _atypes.Float32, _atypes.Float64)
TV_IRFFT2D_Tcomplex = TypeVar("TV_IRFFT2D_Tcomplex", _atypes.Complex128, _atypes.Complex64)

def irfft2d(input: Annotated[Any, TV_IRFFT2D_Tcomplex], fft_length: Annotated[Any, _atypes.Int32], Treal:TV_IRFFT2D_Treal=_dtypes.float32, name=None) -> Annotated[Any, TV_IRFFT2D_Treal]:
  r"""Inverse 2D real-valued fast Fourier transform.

  Computes the inverse 2-dimensional discrete Fourier transform of a real-valued
  signal over the inner-most 2 dimensions of `input`.

  The inner-most 2 dimensions of `input` are assumed to be the result of `RFFT2D`:
  The inner-most dimension contains the `fft_length / 2 + 1` unique components of
  the DFT of a real-valued signal. If `fft_length` is not provided, it is computed
  from the size of the inner-most 2 dimensions of `input`. If the FFT length used
  to compute `input` is odd, it should be provided since it cannot be inferred
  properly.

  Along each axis `IRFFT2D` is computed on, if `fft_length` (or
  `fft_length / 2 + 1` for the inner-most dimension) is smaller than the
  corresponding dimension of `input`, the dimension is cropped. If it is larger,
  the dimension is padded with zeros.

  Args:
    input: A `Tensor`. Must be one of the following types: `complex64`, `complex128`.
      A complex tensor.
    fft_length: A `Tensor` of type `int32`.
      An int32 tensor of shape [2]. The FFT length for each dimension.
    Treal: An optional `tf.DType` from: `tf.float32, tf.float64`. Defaults to `tf.float32`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `Treal`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "IRFFT2D", name, input, fft_length, "Treal", Treal)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return irfft2d_eager_fallback(
          input, fft_length, Treal=Treal, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if Treal is None:
    Treal = _dtypes.float32
  Treal = _execute.make_type(Treal, "Treal")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "IRFFT2D", input=input, fft_length=fft_length, Treal=Treal, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("Treal", _op._get_attr_type("Treal"), "Tcomplex",
              _op._get_attr_type("Tcomplex"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "IRFFT2D", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

IRFFT2D = tf_export("raw_ops.IRFFT2D")(_ops.to_raw_op(irfft2d))


def irfft2d_eager_fallback(input: Annotated[Any, TV_IRFFT2D_Tcomplex], fft_length: Annotated[Any, _atypes.Int32], Treal: TV_IRFFT2D_Treal, name, ctx) -> Annotated[Any, TV_IRFFT2D_Treal]:
  if Treal is None:
    Treal = _dtypes.float32
  Treal = _execute.make_type(Treal, "Treal")
  _attr_Tcomplex, (input,) = _execute.args_to_matching_eager([input], ctx, [_dtypes.complex64, _dtypes.complex128, ], _dtypes.complex64)
  fft_length = _ops.convert_to_tensor(fft_length, _dtypes.int32)
  _inputs_flat = [input, fft_length]
  _attrs = ("Treal", Treal, "Tcomplex", _attr_Tcomplex)
  _result = _execute.execute(b"IRFFT2D", 1, inputs=_inputs_flat, attrs=_attrs,
                             ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "IRFFT2D", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_IRFFT3D_Treal = TypeVar("TV_IRFFT3D_Treal", _atypes.Float32, _atypes.Float64)
TV_IRFFT3D_Tcomplex = TypeVar("TV_IRFFT3D_Tcomplex", _atypes.Complex128, _atypes.Complex64)

def irfft3d(input: Annotated[Any, TV_IRFFT3D_Tcomplex], fft_length: Annotated[Any, _atypes.Int32], Treal:TV_IRFFT3D_Treal=_dtypes.float32, name=None) -> Annotated[Any, TV_IRFFT3D_Treal]:
  r"""Inverse 3D real-valued fast Fourier transform.

  Computes the inverse 3-dimensional discrete Fourier transform of a real-valued
  signal over the inner-most 3 dimensions of `input`.

  The inner-most 3 dimensions of `input` are assumed to be the result of `RFFT3D`:
  The inner-most dimension contains the `fft_length / 2 + 1` unique components of
  the DFT of a real-valued signal. If `fft_length` is not provided, it is computed
  from the size of the inner-most 3 dimensions of `input`. If the FFT length used
  to compute `input` is odd, it should be provided since it cannot be inferred
  properly.

  Along each axis `IRFFT3D` is computed on, if `fft_length` (or
  `fft_length / 2 + 1` for the inner-most dimension) is smaller than the
  corresponding dimension of `input`, the dimension is cropped. If it is larger,
  the dimension is padded with zeros.

  Args:
    input: A `Tensor`. Must be one of the following types: `complex64`, `complex128`.
      A complex tensor.
    fft_length: A `Tensor` of type `int32`.
      An int32 tensor of shape [3]. The FFT length for each dimension.
    Treal: An optional `tf.DType` from: `tf.float32, tf.float64`. Defaults to `tf.float32`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `Treal`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "IRFFT3D", name, input, fft_length, "Treal", Treal)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return irfft3d_eager_fallback(
          input, fft_length, Treal=Treal, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if Treal is None:
    Treal = _dtypes.float32
  Treal = _execute.make_type(Treal, "Treal")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "IRFFT3D", input=input, fft_length=fft_length, Treal=Treal, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("Treal", _op._get_attr_type("Treal"), "Tcomplex",
              _op._get_attr_type("Tcomplex"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "IRFFT3D", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

IRFFT3D = tf_export("raw_ops.IRFFT3D")(_ops.to_raw_op(irfft3d))


def irfft3d_eager_fallback(input: Annotated[Any, TV_IRFFT3D_Tcomplex], fft_length: Annotated[Any, _atypes.Int32], Treal: TV_IRFFT3D_Treal, name, ctx) -> Annotated[Any, TV_IRFFT3D_Treal]:
  if Treal is None:
    Treal = _dtypes.float32
  Treal = _execute.make_type(Treal, "Treal")
  _attr_Tcomplex, (input,) = _execute.args_to_matching_eager([input], ctx, [_dtypes.complex64, _dtypes.complex128, ], _dtypes.complex64)
  fft_length = _ops.convert_to_tensor(fft_length, _dtypes.int32)
  _inputs_flat = [input, fft_length]
  _attrs = ("Treal", Treal, "Tcomplex", _attr_Tcomplex)
  _result = _execute.execute(b"IRFFT3D", 1, inputs=_inputs_flat, attrs=_attrs,
                             ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "IRFFT3D", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_IRFFTND_Treal = TypeVar("TV_IRFFTND_Treal", _atypes.Float32, _atypes.Float64)
TV_IRFFTND_Tcomplex = TypeVar("TV_IRFFTND_Tcomplex", _atypes.Complex128, _atypes.Complex64)

@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('irfftnd')
def irfftnd(input: Annotated[Any, TV_IRFFTND_Tcomplex], fft_length: Annotated[Any, _atypes.Int32], axes: Annotated[Any, _atypes.Int32], Treal:TV_IRFFTND_Treal=_dtypes.float32, name=None) -> Annotated[Any, TV_IRFFTND_Treal]:
  r"""ND inverse real fast Fourier transform.

  Computes the n-dimensional inverse real discrete Fourier transform over
  designated dimensions of `input`. The designated dimensions of `input` are
  assumed to be the result of `IRFFTND`. The inner-most dimension contains the
  `fft_length / 2 + 1` unique components of the DFT of a real-valued signal. 

  If fft_length[i]<shape(input)[i], the input is cropped. If
  fft_length[i]>shape(input)[i], the input is padded with zeros. If fft_length
  is not given, the default shape(input) is used.

  Axes mean the dimensions to perform the transform on. Default is to perform on
  all axes.

  Args:
    input: A `Tensor`. Must be one of the following types: `complex64`, `complex128`.
      A complex tensor.
    fft_length: A `Tensor` of type `int32`.
      An int32 tensor. The FFT length for each dimension.
    axes: A `Tensor` of type `int32`.
      An int32 tensor with a same shape as fft_length. Axes to perform the transform.
    Treal: An optional `tf.DType` from: `tf.float32, tf.float64`. Defaults to `tf.float32`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `Treal`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "IRFFTND", name, input, fft_length, axes, "Treal", Treal)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_irfftnd(
          (input, fft_length, axes, Treal, name,), None)
      if _result is not NotImplemented:
        return _result
      return irfftnd_eager_fallback(
          input, fft_length, axes, Treal=Treal, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            irfftnd, (), dict(input=input, fft_length=fft_length, axes=axes,
                              Treal=Treal, name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_irfftnd(
        (input, fft_length, axes, Treal, name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  if Treal is None:
    Treal = _dtypes.float32
  Treal = _execute.make_type(Treal, "Treal")
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "IRFFTND", input=input, fft_length=fft_length, axes=axes, Treal=Treal,
                   name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          irfftnd, (), dict(input=input, fft_length=fft_length, axes=axes,
                            Treal=Treal, name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("Treal", _op._get_attr_type("Treal"), "Tcomplex",
              _op._get_attr_type("Tcomplex"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "IRFFTND", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

IRFFTND = tf_export("raw_ops.IRFFTND")(_ops.to_raw_op(irfftnd))
_dispatcher_for_irfftnd = irfftnd._tf_type_based_dispatcher.Dispatch


def irfftnd_eager_fallback(input: Annotated[Any, TV_IRFFTND_Tcomplex], fft_length: Annotated[Any, _atypes.Int32], axes: Annotated[Any, _atypes.Int32], Treal: TV_IRFFTND_Treal, name, ctx) -> Annotated[Any, TV_IRFFTND_Treal]:
  if Treal is None:
    Treal = _dtypes.float32
  Treal = _execute.make_type(Treal, "Treal")
  _attr_Tcomplex, (input,) = _execute.args_to_matching_eager([input], ctx, [_dtypes.complex64, _dtypes.complex128, ], _dtypes.complex64)
  fft_length = _ops.convert_to_tensor(fft_length, _dtypes.int32)
  axes = _ops.convert_to_tensor(axes, _dtypes.int32)
  _inputs_flat = [input, fft_length, axes]
  _attrs = ("Treal", Treal, "Tcomplex", _attr_Tcomplex)
  _result = _execute.execute(b"IRFFTND", 1, inputs=_inputs_flat, attrs=_attrs,
                             ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "IRFFTND", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_RFFT_Treal = TypeVar("TV_RFFT_Treal", _atypes.Float32, _atypes.Float64)
TV_RFFT_Tcomplex = TypeVar("TV_RFFT_Tcomplex", _atypes.Complex128, _atypes.Complex64)

def rfft(input: Annotated[Any, TV_RFFT_Treal], fft_length: Annotated[Any, _atypes.Int32], Tcomplex:TV_RFFT_Tcomplex=_dtypes.complex64, name=None) -> Annotated[Any, TV_RFFT_Tcomplex]:
  r"""Real-valued fast Fourier transform.

  Computes the 1-dimensional discrete Fourier transform of a real-valued signal
  over the inner-most dimension of `input`.

  Since the DFT of a real signal is Hermitian-symmetric, `RFFT` only returns the
  `fft_length / 2 + 1` unique components of the FFT: the zero-frequency term,
  followed by the `fft_length / 2` positive-frequency terms.

  Along the axis `RFFT` is computed on, if `fft_length` is smaller than the
  corresponding dimension of `input`, the dimension is cropped. If it is larger,
  the dimension is padded with zeros.

  Args:
    input: A `Tensor`. Must be one of the following types: `float32`, `float64`.
      A float32 tensor.
    fft_length: A `Tensor` of type `int32`.
      An int32 tensor of shape [1]. The FFT length.
    Tcomplex: An optional `tf.DType` from: `tf.complex64, tf.complex128`. Defaults to `tf.complex64`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `Tcomplex`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "RFFT", name, input, fft_length, "Tcomplex", Tcomplex)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return rfft_eager_fallback(
          input, fft_length, Tcomplex=Tcomplex, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if Tcomplex is None:
    Tcomplex = _dtypes.complex64
  Tcomplex = _execute.make_type(Tcomplex, "Tcomplex")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "RFFT", input=input, fft_length=fft_length, Tcomplex=Tcomplex,
                name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("Treal", _op._get_attr_type("Treal"), "Tcomplex",
              _op._get_attr_type("Tcomplex"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "RFFT", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

RFFT = tf_export("raw_ops.RFFT")(_ops.to_raw_op(rfft))


def rfft_eager_fallback(input: Annotated[Any, TV_RFFT_Treal], fft_length: Annotated[Any, _atypes.Int32], Tcomplex: TV_RFFT_Tcomplex, name, ctx) -> Annotated[Any, TV_RFFT_Tcomplex]:
  if Tcomplex is None:
    Tcomplex = _dtypes.complex64
  Tcomplex = _execute.make_type(Tcomplex, "Tcomplex")
  _attr_Treal, (input,) = _execute.args_to_matching_eager([input], ctx, [_dtypes.float32, _dtypes.float64, ], _dtypes.float32)
  fft_length = _ops.convert_to_tensor(fft_length, _dtypes.int32)
  _inputs_flat = [input, fft_length]
  _attrs = ("Treal", _attr_Treal, "Tcomplex", Tcomplex)
  _result = _execute.execute(b"RFFT", 1, inputs=_inputs_flat, attrs=_attrs,
                             ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "RFFT", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_RFFT2D_Treal = TypeVar("TV_RFFT2D_Treal", _atypes.Float32, _atypes.Float64)
TV_RFFT2D_Tcomplex = TypeVar("TV_RFFT2D_Tcomplex", _atypes.Complex128, _atypes.Complex64)

def rfft2d(input: Annotated[Any, TV_RFFT2D_Treal], fft_length: Annotated[Any, _atypes.Int32], Tcomplex:TV_RFFT2D_Tcomplex=_dtypes.complex64, name=None) -> Annotated[Any, TV_RFFT2D_Tcomplex]:
  r"""2D real-valued fast Fourier transform.

  Computes the 2-dimensional discrete Fourier transform of a real-valued signal
  over the inner-most 2 dimensions of `input`.

  Since the DFT of a real signal is Hermitian-symmetric, `RFFT2D` only returns the
  `fft_length / 2 + 1` unique components of the FFT for the inner-most dimension
  of `output`: the zero-frequency term, followed by the `fft_length / 2`
  positive-frequency terms.

  Along each axis `RFFT2D` is computed on, if `fft_length` is smaller than the
  corresponding dimension of `input`, the dimension is cropped. If it is larger,
  the dimension is padded with zeros.

  Args:
    input: A `Tensor`. Must be one of the following types: `float32`, `float64`.
      A float32 tensor.
    fft_length: A `Tensor` of type `int32`.
      An int32 tensor of shape [2]. The FFT length for each dimension.
    Tcomplex: An optional `tf.DType` from: `tf.complex64, tf.complex128`. Defaults to `tf.complex64`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `Tcomplex`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "RFFT2D", name, input, fft_length, "Tcomplex", Tcomplex)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return rfft2d_eager_fallback(
          input, fft_length, Tcomplex=Tcomplex, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if Tcomplex is None:
    Tcomplex = _dtypes.complex64
  Tcomplex = _execute.make_type(Tcomplex, "Tcomplex")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "RFFT2D", input=input, fft_length=fft_length, Tcomplex=Tcomplex,
                  name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("Treal", _op._get_attr_type("Treal"), "Tcomplex",
              _op._get_attr_type("Tcomplex"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "RFFT2D", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

RFFT2D = tf_export("raw_ops.RFFT2D")(_ops.to_raw_op(rfft2d))


def rfft2d_eager_fallback(input: Annotated[Any, TV_RFFT2D_Treal], fft_length: Annotated[Any, _atypes.Int32], Tcomplex: TV_RFFT2D_Tcomplex, name, ctx) -> Annotated[Any, TV_RFFT2D_Tcomplex]:
  if Tcomplex is None:
    Tcomplex = _dtypes.complex64
  Tcomplex = _execute.make_type(Tcomplex, "Tcomplex")
  _attr_Treal, (input,) = _execute.args_to_matching_eager([input], ctx, [_dtypes.float32, _dtypes.float64, ], _dtypes.float32)
  fft_length = _ops.convert_to_tensor(fft_length, _dtypes.int32)
  _inputs_flat = [input, fft_length]
  _attrs = ("Treal", _attr_Treal, "Tcomplex", Tcomplex)
  _result = _execute.execute(b"RFFT2D", 1, inputs=_inputs_flat, attrs=_attrs,
                             ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "RFFT2D", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_RFFT3D_Treal = TypeVar("TV_RFFT3D_Treal", _atypes.Float32, _atypes.Float64)
TV_RFFT3D_Tcomplex = TypeVar("TV_RFFT3D_Tcomplex", _atypes.Complex128, _atypes.Complex64)

def rfft3d(input: Annotated[Any, TV_RFFT3D_Treal], fft_length: Annotated[Any, _atypes.Int32], Tcomplex:TV_RFFT3D_Tcomplex=_dtypes.complex64, name=None) -> Annotated[Any, TV_RFFT3D_Tcomplex]:
  r"""3D real-valued fast Fourier transform.

  Computes the 3-dimensional discrete Fourier transform of a real-valued signal
  over the inner-most 3 dimensions of `input`.

  Since the DFT of a real signal is Hermitian-symmetric, `RFFT3D` only returns the
  `fft_length / 2 + 1` unique components of the FFT for the inner-most dimension
  of `output`: the zero-frequency term, followed by the `fft_length / 2`
  positive-frequency terms.

  Along each axis `RFFT3D` is computed on, if `fft_length` is smaller than the
  corresponding dimension of `input`, the dimension is cropped. If it is larger,
  the dimension is padded with zeros.

  Args:
    input: A `Tensor`. Must be one of the following types: `float32`, `float64`.
      A float32 tensor.
    fft_length: A `Tensor` of type `int32`.
      An int32 tensor of shape [3]. The FFT length for each dimension.
    Tcomplex: An optional `tf.DType` from: `tf.complex64, tf.complex128`. Defaults to `tf.complex64`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `Tcomplex`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "RFFT3D", name, input, fft_length, "Tcomplex", Tcomplex)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return rfft3d_eager_fallback(
          input, fft_length, Tcomplex=Tcomplex, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if Tcomplex is None:
    Tcomplex = _dtypes.complex64
  Tcomplex = _execute.make_type(Tcomplex, "Tcomplex")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "RFFT3D", input=input, fft_length=fft_length, Tcomplex=Tcomplex,
                  name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("Treal", _op._get_attr_type("Treal"), "Tcomplex",
              _op._get_attr_type("Tcomplex"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "RFFT3D", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

RFFT3D = tf_export("raw_ops.RFFT3D")(_ops.to_raw_op(rfft3d))


def rfft3d_eager_fallback(input: Annotated[Any, TV_RFFT3D_Treal], fft_length: Annotated[Any, _atypes.Int32], Tcomplex: TV_RFFT3D_Tcomplex, name, ctx) -> Annotated[Any, TV_RFFT3D_Tcomplex]:
  if Tcomplex is None:
    Tcomplex = _dtypes.complex64
  Tcomplex = _execute.make_type(Tcomplex, "Tcomplex")
  _attr_Treal, (input,) = _execute.args_to_matching_eager([input], ctx, [_dtypes.float32, _dtypes.float64, ], _dtypes.float32)
  fft_length = _ops.convert_to_tensor(fft_length, _dtypes.int32)
  _inputs_flat = [input, fft_length]
  _attrs = ("Treal", _attr_Treal, "Tcomplex", Tcomplex)
  _result = _execute.execute(b"RFFT3D", 1, inputs=_inputs_flat, attrs=_attrs,
                             ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "RFFT3D", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


TV_RFFTND_Treal = TypeVar("TV_RFFTND_Treal", _atypes.Float32, _atypes.Float64)
TV_RFFTND_Tcomplex = TypeVar("TV_RFFTND_Tcomplex", _atypes.Complex128, _atypes.Complex64)

@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('rfftnd')
def rfftnd(input: Annotated[Any, TV_RFFTND_Treal], fft_length: Annotated[Any, _atypes.Int32], axes: Annotated[Any, _atypes.Int32], Tcomplex:TV_RFFTND_Tcomplex=_dtypes.complex64, name=None) -> Annotated[Any, TV_RFFTND_Tcomplex]:
  r"""ND fast real Fourier transform.

  Computes the n-dimensional real discrete Fourier transform over designated
  dimensions of `input`. The designated dimensions of `input` are assumed to be
  the result of `RFFTND`. The length of the last axis transformed will be
  fft_length[-1]//2+1.

  If fft_length[i]<shape(input)[i], the input is cropped. If
  fft_length[i]>shape(input)[i], the input is padded with zeros. If fft_length
  is not given, the default shape(input) is used.

  Axes mean the dimensions to perform the transform on. Default is to perform on
  all axes.

  Args:
    input: A `Tensor`. Must be one of the following types: `float32`, `float64`.
      A complex tensor.
    fft_length: A `Tensor` of type `int32`.
      An int32 tensor. The FFT length for each dimension.
    axes: A `Tensor` of type `int32`.
      An int32 tensor with a same shape as fft_length. Axes to perform the transform.
    Tcomplex: An optional `tf.DType` from: `tf.complex64, tf.complex128`. Defaults to `tf.complex64`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `Tcomplex`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "RFFTND", name, input, fft_length, axes, "Tcomplex", Tcomplex)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_rfftnd(
          (input, fft_length, axes, Tcomplex, name,), None)
      if _result is not NotImplemented:
        return _result
      return rfftnd_eager_fallback(
          input, fft_length, axes, Tcomplex=Tcomplex, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            rfftnd, (), dict(input=input, fft_length=fft_length, axes=axes,
                             Tcomplex=Tcomplex, name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_rfftnd(
        (input, fft_length, axes, Tcomplex, name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  if Tcomplex is None:
    Tcomplex = _dtypes.complex64
  Tcomplex = _execute.make_type(Tcomplex, "Tcomplex")
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "RFFTND", input=input, fft_length=fft_length, axes=axes,
                  Tcomplex=Tcomplex, name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          rfftnd, (), dict(input=input, fft_length=fft_length, axes=axes,
                           Tcomplex=Tcomplex, name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("Treal", _op._get_attr_type("Treal"), "Tcomplex",
              _op._get_attr_type("Tcomplex"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "RFFTND", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

RFFTND = tf_export("raw_ops.RFFTND")(_ops.to_raw_op(rfftnd))
_dispatcher_for_rfftnd = rfftnd._tf_type_based_dispatcher.Dispatch


def rfftnd_eager_fallback(input: Annotated[Any, TV_RFFTND_Treal], fft_length: Annotated[Any, _atypes.Int32], axes: Annotated[Any, _atypes.Int32], Tcomplex: TV_RFFTND_Tcomplex, name, ctx) -> Annotated[Any, TV_RFFTND_Tcomplex]:
  if Tcomplex is None:
    Tcomplex = _dtypes.complex64
  Tcomplex = _execute.make_type(Tcomplex, "Tcomplex")
  _attr_Treal, (input,) = _execute.args_to_matching_eager([input], ctx, [_dtypes.float32, _dtypes.float64, ], _dtypes.float32)
  fft_length = _ops.convert_to_tensor(fft_length, _dtypes.int32)
  axes = _ops.convert_to_tensor(axes, _dtypes.int32)
  _inputs_flat = [input, fft_length, axes]
  _attrs = ("Treal", _attr_Treal, "Tcomplex", Tcomplex)
  _result = _execute.execute(b"RFFTND", 1, inputs=_inputs_flat, attrs=_attrs,
                             ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "RFFTND", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

