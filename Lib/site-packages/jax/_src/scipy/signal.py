# Copyright 2020 The JAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from collections.abc import Callable, Sequence
from functools import partial
import math
import operator
from typing import Any
import warnings

import numpy as np

import jax
import jax.numpy.fft
import jax.numpy as jnp
from jax import lax
from jax._src import core
from jax._src import dtypes
from jax._src.api_util import _ensure_index_tuple
from jax._src.lax.lax import PrecisionLike
from jax._src.numpy import linalg
from jax._src.numpy.util import (
    check_arraylike, promote_dtypes_inexact, promote_dtypes_complex)
from jax._src.third_party.scipy import signal_helper
from jax._src.typing import Array, ArrayLike
from jax._src.util import canonicalize_axis, tuple_delete, tuple_insert


def fftconvolve(in1: ArrayLike, in2: ArrayLike, mode: str = "full",
                axes: Sequence[int] | None = None) -> Array:
  """
  Convolve two N-dimensional arrays using Fast Fourier Transform (FFT).

  JAX implementation of :func:`scipy.signal.fftconvolve`.

  Args:
    in1: left-hand input to the convolution.
    in2: right-hand input to the convolution. Must have ``in1.ndim == in2.ndim``.
    mode: controls the size of the output. Available operations are:

      * ``"full"``: (default) output the full convolution of the inputs.
      * ``"same"``: return a centered portion of the ``"full"`` output which
        is the same size as ``in1``.
      * ``"valid"``: return the portion of the ``"full"`` output which do not
        depend on padding at the array edges.

    axes: optional sequence of axes along which to apply the convolution.

  Returns:
    Array containing the convolved result.

  See Also:
    - :func:`jax.numpy.convolve`: 1D convolution
    - :func:`jax.scipy.signal.convolve`: direct convolution

  Examples:
    A few 1D convolution examples. Because FFT-based convolution is approximate,
    We use :func:`jax.numpy.printoptions` below to adjust the printing precision:

    >>> x = jnp.array([1, 2, 3, 2, 1])
    >>> y = jnp.array([1, 1, 1])

    Full convolution uses implicit zero-padding at the edges:

    >>> with jax.numpy.printoptions(precision=3):
    ...   print(jax.scipy.signal.fftconvolve(x, y, mode='full'))
    [1. 3. 6. 7. 6. 3. 1.]

    Specifying ``mode = 'same'`` returns a centered convolution the same size
    as the first input:

    >>> with jax.numpy.printoptions(precision=3):
    ...   print(jax.scipy.signal.fftconvolve(x, y, mode='same'))
    [3. 6. 7. 6. 3.]

    Specifying ``mode = 'valid'`` returns only the portion where the two arrays
    fully overlap:

    >>> with jax.numpy.printoptions(precision=3):
    ...   print(jax.scipy.signal.fftconvolve(x, y, mode='valid'))
    [6. 7. 6.]
  """
  check_arraylike('fftconvolve', in1, in2)
  in1, in2 = promote_dtypes_inexact(in1, in2)
  if in1.ndim != in2.ndim:
    raise ValueError("in1 and in2 should have the same dimensionality")
  if mode not in ["same", "full", "valid"]:
    raise ValueError("mode must be one of ['same', 'full', 'valid']")
  _fftconvolve = partial(_fftconvolve_unbatched, mode=mode)
  if axes is None:
    return _fftconvolve(in1, in2)
  axes = _ensure_index_tuple(axes)
  axes = tuple(canonicalize_axis(ax, in1.ndim) for ax in axes)
  mapped_axes = set(range(in1.ndim)) - set(axes)
  if any(in1.shape[i] != in2.shape[i] for i in mapped_axes):
    raise ValueError(f"mapped axes must have same shape; got {in1.shape=} {in2.shape=} {axes=}")
  for ax in sorted(mapped_axes):
    _fftconvolve = jax.vmap(_fftconvolve, in_axes=ax, out_axes=ax)
  return _fftconvolve(in1, in2)

def _fftconvolve_unbatched(in1: Array, in2: Array, mode: str) -> Array:
  full_shape = tuple(s1 + s2 - 1 for s1, s2 in zip(in1.shape, in2.shape))

  # TODO(jakevdp): potentially use next_fast_len to evaluate with a more efficient shape.
  fft_shape = full_shape  # tuple(next_fast_len(s) for s in full_shape)

  if mode == 'valid':
    no_swap = all(s1 >= s2 for s1, s2 in zip(in1.shape, in2.shape))
    swap = all(s1 <= s2 for s1, s2 in zip(in1.shape, in2.shape))
    if not (no_swap or swap):
      raise ValueError("For 'valid' mode, One input must be at least as "
                       "large as the other in every dimension.")
    if swap:
      in1, in2 = in2, in1

  if jnp.iscomplexobj(in1):
    fft, ifft = jnp.fft.fftn, jnp.fft.ifftn
  else:
    fft, ifft = jnp.fft.rfftn, jnp.fft.irfftn
  sp1 = fft(in1, fft_shape)
  sp2 = fft(in2, fft_shape)
  conv = ifft(sp1 * sp2, fft_shape)

  if mode == "full":
    out_shape = full_shape
  elif mode == "same":
    out_shape = in1.shape
  elif mode == "valid":
    out_shape = tuple(s1 - s2 + 1 for s1, s2 in zip(in1.shape, in2.shape))
  else:
    raise ValueError(f"Unrecognized {mode=}")

  start_indices = tuple((full_size - out_size) // 2
                        for full_size, out_size in zip(full_shape, out_shape))
  return lax.dynamic_slice(conv, start_indices, out_shape)


# Note: we do not re-use the code from jax.numpy.convolve here, because the handling
# of padding differs slightly between the two implementations (particularly for
# mode='same').
def _convolve_nd(in1: Array, in2: Array, mode: str, *, precision: PrecisionLike) -> Array:
  if mode not in ["full", "same", "valid"]:
    raise ValueError("mode must be one of ['full', 'same', 'valid']")
  if in1.ndim != in2.ndim:
    raise ValueError("in1 and in2 must have the same number of dimensions")
  if in1.size == 0 or in2.size == 0:
    raise ValueError(f"zero-size arrays not supported in convolutions, got shapes {in1.shape} and {in2.shape}.")
  in1, in2 = promote_dtypes_inexact(in1, in2)

  no_swap = all(s1 >= s2 for s1, s2 in zip(in1.shape, in2.shape))
  swap = all(s1 <= s2 for s1, s2 in zip(in1.shape, in2.shape))
  if not (no_swap or swap):
    raise ValueError("One input must be smaller than the other in every dimension.")

  shape_o = in2.shape
  if swap:
    in1, in2 = in2, in1
  shape = in2.shape
  in2 = jnp.flip(in2)

  if mode == 'valid':
    padding = [(0, 0) for s in shape]
  elif mode == 'same':
    padding = [(s - 1 - (s_o - 1) // 2, s - s_o + (s_o - 1) // 2)
               for (s, s_o) in zip(shape, shape_o)]
  elif mode == 'full':
    padding = [(s - 1, s - 1) for s in shape]

  strides = tuple(1 for s in shape)
  result = lax.conv_general_dilated(in1[None, None], in2[None, None], strides,
                                    padding, precision=precision)
  return result[0, 0]


def convolve(in1: Array, in2: Array, mode: str = 'full', method: str = 'auto',
             precision: PrecisionLike = None) -> Array:
  """Convolution of two N-dimensional arrays.

  JAX implementation of :func:`scipy.signal.convolve`.

  Args:
    in1: left-hand input to the convolution.
    in2: right-hand input to the convolution. Must have ``in1.ndim == in2.ndim``.
    mode: controls the size of the output. Available operations are:

      * ``"full"``: (default) output the full convolution of the inputs.
      * ``"same"``: return a centered portion of the ``"full"`` output which
        is the same size as ``in1``.
      * ``"valid"``: return the portion of the ``"full"`` output which do not
        depend on padding at the array edges.

    method: controls the computation method. Options are

      * ``"auto"``: (default) always uses the ``"direct"`` method.
      * ``"direct"``: lower to :func:`jax.lax.conv_general_dilated`.
      * ``"fft"``: compute the result via a fast Fourier transform.

    precision: Specify the precision of the computation. Refer to
      :class:`jax.lax.Precision` for a description of available values.

  Returns:
    Array containing the convolved result.

  See Also:
    - :func:`jax.numpy.convolve`: 1D convolution
    - :func:`jax.scipy.signal.convolve2d`: 2D convolution
    - :func:`jax.scipy.signal.correlate`: ND correlation

  Examples:
    A few 1D convolution examples:

    >>> x = jnp.array([1, 2, 3, 2, 1])
    >>> y = jnp.array([1, 1, 1])

    Full convolution uses implicit zero-padding at the edges:

    >>> jax.scipy.signal.convolve(x, y, mode='full')
    Array([1., 3., 6., 7., 6., 3., 1.], dtype=float32)

    Specifying ``mode = 'same'`` returns a centered convolution the same size
    as the first input:

    >>> jax.scipy.signal.convolve(x, y, mode='same')
    Array([3., 6., 7., 6., 3.], dtype=float32)

    Specifying ``mode = 'valid'`` returns only the portion where the two arrays
    fully overlap:

    >>> jax.scipy.signal.convolve(x, y, mode='valid')
    Array([6., 7., 6.], dtype=float32)
  """
  if method == 'fft':
    return fftconvolve(in1, in2, mode=mode)
  elif method in ['direct', 'auto']:
    return _convolve_nd(in1, in2, mode, precision=precision)
  else:
    raise ValueError(f"Got {method=}; expected 'auto', 'fft', or 'direct'.")


def convolve2d(in1: Array, in2: Array, mode: str = 'full', boundary: str = 'fill',
               fillvalue: float = 0, precision: PrecisionLike = None) -> Array:
  """Convolution of two 2-dimensional arrays.

  JAX implementation of :func:`scipy.signal.convolve2d`.

  Args:
    in1: left-hand input to the convolution. Must have ``in1.ndim == 2``.
    in2: right-hand input to the convolution. Must have ``in2.ndim == 2``.
    mode: controls the size of the output. Available operations are:

      * ``"full"``: (default) output the full convolution of the inputs.
      * ``"same"``: return a centered portion of the ``"full"`` output which
        is the same size as ``in1``.
      * ``"valid"``: return the portion of the ``"full"`` output which do not
        depend on padding at the array edges.

    boundary: only ``"fill"`` is supported.
    fillvalue: only ``0`` is supported.
    method: controls the computation method. Options are

      * ``"auto"``: (default) always uses the ``"direct"`` method.
      * ``"direct"``: lower to :func:`jax.lax.conv_general_dilated`.
      * ``"fft"``: compute the result via a fast Fourier transform.

    precision: Specify the precision of the computation. Refer to
      :class:`jax.lax.Precision` for a description of available values.

  Returns:
    Array containing the convolved result.

  See Also:
    - :func:`jax.numpy.convolve`: 1D convolution
    - :func:`jax.scipy.signal.convolve`: ND convolution
    - :func:`jax.scipy.signal.correlate`: ND correlation

  Examples:
    A few 2D convolution examples:

    >>> x = jnp.array([[1, 2],
    ...                [3, 4]])
    >>> y = jnp.array([[2, 1, 1],
    ...                [4, 3, 4],
    ...                [1, 3, 2]])

    Full 2D convolution uses implicit zero-padding at the edges:

    >>> jax.scipy.signal.convolve2d(x, y, mode='full')
    Array([[ 2.,  5.,  3.,  2.],
           [10., 22., 17., 12.],
           [13., 30., 32., 20.],
           [ 3., 13., 18.,  8.]], dtype=float32)

    Specifying ``mode = 'same'`` returns a centered 2D convolution of the same size
    as the first input:

    >>> jax.scipy.signal.convolve2d(x, y, mode='same')
    Array([[22., 17.],
           [30., 32.]], dtype=float32)

    Specifying ``mode = 'valid'`` returns only the portion of 2D convolution
    where the two arrays fully overlap:

    >>> jax.scipy.signal.convolve2d(x, y, mode='valid')
    Array([[22., 17.],
           [30., 32.]], dtype=float32)
  """
  if boundary != 'fill' or fillvalue != 0:
    raise NotImplementedError("convolve2d() only supports boundary='fill', fillvalue=0")
  if jnp.ndim(in1) != 2 or jnp.ndim(in2) != 2:
    raise ValueError("convolve2d() only supports 2-dimensional inputs.")
  return _convolve_nd(in1, in2, mode, precision=precision)


def correlate(in1: Array, in2: Array, mode: str = 'full', method: str = 'auto',
              precision: PrecisionLike = None) -> Array:
  """Cross-correlation of two N-dimensional arrays.

  JAX implementation of :func:`scipy.signal.correlate`.

  Args:
    in1: left-hand input to the cross-correlation.
    in2: right-hand input to the cross-correlation. Must have ``in1.ndim == in2.ndim``.
    mode: controls the size of the output. Available operations are:

      * ``"full"``: (default) output the full cross-correlation of the inputs.
      * ``"same"``: return a centered portion of the ``"full"`` output which
        is the same size as ``in1``.
      * ``"valid"``: return the portion of the ``"full"`` output which do not
        depend on padding at the array edges.

    method: controls the computation method. Options are

      * ``"auto"``: (default) always uses the ``"direct"`` method.
      * ``"direct"``: lower to :func:`jax.lax.conv_general_dilated`.
      * ``"fft"``: compute the result via a fast Fourier transform.

    precision: Specify the precision of the computation. Refer to
      :class:`jax.lax.Precision` for a description of available values.

  Returns:
    Array containing the cross-correlation result.

  See Also:
    - :func:`jax.numpy.correlate`: 1D cross-correlation
    - :func:`jax.scipy.signal.correlate2d`: 2D cross-correlation
    - :func:`jax.scipy.signal.convolve`: ND convolution

  Examples:
    A few 1D correlation examples:

    >>> x = jnp.array([1, 2, 3, 2, 1])
    >>> y = jnp.array([1, 3, 2])

    Full 1D correlation uses implicit zero-padding at the edges:

    >>> jax.scipy.signal.correlate(x, y, mode='full')
    Array([ 2.,  7., 13., 15., 11.,  5.,  1.], dtype=float32)

    Specifying ``mode = 'same'`` returns a centered 1D correlation of the same
    size as the first input:

    >>> jax.scipy.signal.correlate(x, y, mode='same')
    Array([ 7., 13., 15., 11.,  5.], dtype=float32)

    Specifying ``mode = 'valid'`` returns only the portion of 1D correlation
    where the two arrays fully overlap:

    >>> jax.scipy.signal.correlate(x, y, mode='valid')
    Array([13., 15., 11.], dtype=float32)
  """
  return convolve(in1, jnp.flip(in2.conj()), mode, precision=precision, method=method)


def correlate2d(in1: Array, in2: Array, mode: str = 'full', boundary: str = 'fill',
                fillvalue: float = 0, precision: PrecisionLike = None) -> Array:
  """Cross-correlation of two 2-dimensional arrays.

  JAX implementation of :func:`scipy.signal.correlate2d`.

  Args:
    in1: left-hand input to the cross-correlation. Must have ``in1.ndim == 2``.
    in2: right-hand input to the cross-correlation. Must have ``in2.ndim == 2``.
    mode: controls the size of the output. Available operations are:

      * ``"full"``: (default) output the full cross-correlation of the inputs.
      * ``"same"``: return a centered portion of the ``"full"`` output which
        is the same size as ``in1``.
      * ``"valid"``: return the portion of the ``"full"`` output which do not
        depend on padding at the array edges.

    boundary: only ``"fill"`` is supported.
    fillvalue: only ``0`` is supported.
    method: controls the computation method. Options are

      * ``"auto"``: (default) always uses the ``"direct"`` method.
      * ``"direct"``: lower to :func:`jax.lax.conv_general_dilated`.
      * ``"fft"``: compute the result via a fast Fourier transform.

    precision: Specify the precision of the computation. Refer to
      :class:`jax.lax.Precision` for a description of available values.

  Returns:
    Array containing the cross-correlation result.

  See Also:
    - :func:`jax.numpy.correlate`: 1D cross-correlation
    - :func:`jax.scipy.signal.correlate`: ND cross-correlation
    - :func:`jax.scipy.signal.convolve`: ND convolution

  Examples:
    A few 2D correlation examples:

    >>> x = jnp.array([[2, 1, 3],
    ...                [1, 3, 1],
    ...                [4, 1, 2]])
    >>> y = jnp.array([[1, 3],
    ...                [4, 2]])

    Full 2D correlation uses implicit zero-padding at the edges:

    >>> jax.scipy.signal.correlate2d(x, y, mode='full')
    Array([[ 4., 10., 10., 12.],
           [ 8., 15., 24.,  7.],
           [11., 28., 14.,  9.],
           [12.,  7.,  7.,  2.]], dtype=float32)

    Specifying ``mode = 'same'`` returns a centered 2D correlation of the same
    size as the first input:

    >>> jax.scipy.signal.correlate2d(x, y, mode='same')
    Array([[15., 24.,  7.],
           [28., 14.,  9.],
           [ 7.,  7.,  2.]], dtype=float32)

    Specifying ``mode = 'valid'`` returns only the portion of 2D correlation
    where the two arrays fully overlap:

    >>> jax.scipy.signal.correlate2d(x, y, mode='valid')
    Array([[15., 24.],
           [28., 14.]], dtype=float32)
  """
  if boundary != 'fill' or fillvalue != 0:
    raise NotImplementedError("correlate2d() only supports boundary='fill', fillvalue=0")
  if jnp.ndim(in1) != 2 or jnp.ndim(in2) != 2:
    raise ValueError("correlate2d() only supports 2-dimensional inputs.")

  swap = all(s1 <= s2 for s1, s2 in zip(in1.shape, in2.shape))
  same_shape =  all(s1 == s2 for s1, s2 in zip(in1.shape, in2.shape))

  if mode == "same":
    in1, in2 = jnp.flip(in1), in2.conj()
    result = jnp.flip(_convolve_nd(in1, in2, mode, precision=precision))
  elif mode == "valid":
    if swap and not same_shape:
      in1, in2 = jnp.flip(in2), in1.conj()
      result = _convolve_nd(in1, in2, mode, precision=precision)
    else:
      in1, in2 = jnp.flip(in1), in2.conj()
      result = jnp.flip(_convolve_nd(in1, in2, mode, precision=precision))
  else:
    if swap:
      in1, in2 = jnp.flip(in2), in1.conj()
      result = _convolve_nd(in1, in2, mode, precision=precision).conj()
    else:
      in1, in2 = jnp.flip(in1), in2.conj()
      result = jnp.flip(_convolve_nd(in1, in2, mode, precision=precision))
  return result


def detrend(data: ArrayLike, axis: int = -1, type: str = 'linear', bp: int = 0,
            overwrite_data: None = None) -> Array:
  """
  Remove linear or piecewise linear trends from data.

  JAX implementation of :func:`scipy.signal.detrend`.

  Args:
    data: The input array containing the data to detrend.
    axis: The axis along which to detrend. Default is -1 (the last axis).
    type: The type of detrending. Can be:

      * ``'linear'``: Fit a single linear trend for the entire data.
      * ``'constant'``: Remove the mean value of the data.

    bp: A sequence of breakpoints. If given, piecewise linear trends
      are fit between these breakpoints.
    overwrite_data: This argument is not supported by JAX's implementation.

  Returns:
    The detrended data array.

  Examples:
    A simple detrend operation in one dimension:

    >>> data = jnp.array([1., 4., 8., 8., 9.])

    Removing a linear trend from the data:

    >>> detrended = jax.scipy.signal.detrend(data)
    >>> with jnp.printoptions(precision=3, suppress=True):  # suppress float error
    ...   print("Detrended:", detrended)
    ...   print("Underlying trend:", data - detrended)
    Detrended: [-1. -0.  2. -0. -1.]
    Underlying trend: [ 2.  4.  6.  8. 10.]

    Removing a constant trend from the data:

    >>> detrended = jax.scipy.signal.detrend(data, type='constant')
    >>> with jnp.printoptions(precision=3):  # suppress float error
    ...   print("Detrended:", detrended)
    ...   print("Underlying trend:", data - detrended)
    Detrended: [-5. -2.  2.  2.  3.]
    Underlying trend: [6. 6. 6. 6. 6.]
  """
  if overwrite_data is not None:
    raise NotImplementedError("overwrite_data argument not implemented.")
  if type not in ['constant', 'linear']:
    raise ValueError("Trend type must be 'linear' or 'constant'.")
  data_arr, = promote_dtypes_inexact(jnp.asarray(data))
  if type == 'constant':
    return data_arr - data_arr.mean(axis, keepdims=True)
  else:
    N = data_arr.shape[axis]
    # bp is static, so we use np operations to avoid pushing to device.
    bp_arr = np.sort(np.unique(np.r_[0, bp, N]))
    if bp_arr[0] < 0 or bp_arr[-1] > N:
      raise ValueError("Breakpoints must be non-negative and less than length of data along given axis.")
    data_arr = jnp.moveaxis(data_arr, axis, 0)
    shape = data_arr.shape
    data_arr = data_arr.reshape(N, -1)
    for m in range(len(bp_arr) - 1):
      Npts = bp_arr[m + 1] - bp_arr[m]
      A = jnp.vstack([
        jnp.ones(Npts, dtype=data_arr.dtype),
        jnp.arange(1, Npts + 1, dtype=data_arr.dtype) / Npts.astype(data_arr.dtype)
      ]).T
      sl = slice(bp_arr[m], bp_arr[m + 1])
      coef, *_ = linalg.lstsq(A, data_arr[sl])
      data_arr = data_arr.at[sl].add(-jnp.matmul(A, coef, precision=lax.Precision.HIGHEST))
    return jnp.moveaxis(data_arr.reshape(shape), 0, axis)


def _fft_helper(x: Array, win: Array, detrend_func: Callable[[Array], Array],
                nperseg: int, noverlap: int, nfft: int | None, sides: str) -> Array:
  """Calculate windowed FFT in the same way the original SciPy does.
  """
  if x.dtype.kind == 'i':
    x = x.astype(win.dtype)

  *batch_shape, signal_length = x.shape
  # Created strided array of data segments
  if nperseg == 1 and noverlap == 0:
    result = x[..., np.newaxis]
  else:
    step = nperseg - noverlap
    batch_shape = list(batch_shape)
    x = x.reshape((math.prod(batch_shape), signal_length, 1))
    result = jax.lax.conv_general_dilated_patches(
        x, (nperseg,), (step,),
        'VALID',
        dimension_numbers=('NTC', 'OIT', 'NTC'))
    result = result.reshape(*batch_shape, *result.shape[-2:])

  # Detrend each data segment individually
  result = detrend_func(result)

  # Apply window by multiplication
  if jnp.iscomplexobj(win):
    result, = promote_dtypes_complex(result)
  result = win.reshape((1,) * len(batch_shape) + (1, nperseg)) * result

  # Perform the fft on last axis. Zero-pads automatically
  if sides == 'twosided':
    return jax.numpy.fft.fft(result, n=nfft)
  else:
    return jax.numpy.fft.rfft(result.real, n=nfft)


def odd_ext(x: Array, n: int, axis: int = -1) -> Array:
  """Extends `x` along with `axis` by odd-extension.

  This function was previously a part of "scipy.signal.signaltools" but is no
  longer exposed.

  Args:
    x : input array
    n : the number of points to be added to the both end
    axis: the axis to be extended
  """
  if n < 1:
    return x
  if n > x.shape[axis] - 1:
    raise ValueError(
        f"The extension length n ({n}) is too big. "
        f"It must not exceed x.shape[axis]-1, which is {x.shape[axis] - 1}.")
  left_end = lax.slice_in_dim(x, 0, 1, axis=axis)
  left_ext = jnp.flip(lax.slice_in_dim(x, 1, n + 1, axis=axis), axis=axis)
  right_end = lax.slice_in_dim(x, -1, None, axis=axis)
  right_ext = jnp.flip(lax.slice_in_dim(x, -(n + 1), -1, axis=axis), axis=axis)
  ext = jnp.concatenate((2 * left_end - left_ext,
                         x,
                         2 * right_end - right_ext),
                         axis=axis)
  return ext


def _spectral_helper(x: Array, y: ArrayLike | None, fs: ArrayLike = 1.0,
                     window: str = 'hann', nperseg: int | None = None,
                     noverlap: int | None = None, nfft: int | None = None,
                     detrend_type: bool | str | Callable[[Array], Array] = 'constant',
                     return_onesided: bool = True, scaling: str = 'density',
                     axis: int = -1, mode: str = 'psd', boundary: str | None = None,
                     padded: bool = False) -> tuple[Array, Array, Array]:
  """LAX-backend implementation of `scipy.signal._spectral_helper`.

  Unlike the original helper function, `y` can be None for explicitly
  indicating auto-spectral (non cross-spectral) computation.  In addition to
  this, `detrend` argument is renamed to `detrend_type` for avoiding internal
  name overlap.
  """
  if mode not in ('psd', 'stft'):
    raise ValueError(f"Unknown value for mode {mode}, "
                     "must be one of: ('psd', 'stft')")

  def make_pad(mode, **kwargs):
    def pad(x, n, axis=-1):
      pad_width = [(0, 0) for unused_n in range(x.ndim)]
      pad_width[axis] = (n, n)
      return jnp.pad(x, pad_width, mode, **kwargs)
    return pad

  boundary_funcs = {
      'even': make_pad('reflect'),
      'odd': odd_ext,
      'constant': make_pad('edge'),
      'zeros': make_pad('constant', constant_values=0.0),
      None: lambda x, *args, **kwargs: x
  }

  # Check/ normalize inputs
  if boundary not in boundary_funcs:
    raise ValueError(
        f"Unknown boundary option '{boundary}', "
        f"must be one of: {list(boundary_funcs.keys())}")

  axis = core.concrete_or_error(operator.index, axis, "axis of windowed-FFT")
  axis = canonicalize_axis(axis, x.ndim)

  if y is None:
    check_arraylike('spectral_helper', x)
    x, = promote_dtypes_inexact(x)
    y_arr = x  # place-holder for type checking
    outershape = tuple_delete(x.shape, axis)
  else:
    if mode != 'psd':
      raise ValueError("two-argument mode is available only when mode=='psd'")
    check_arraylike('spectral_helper', x, y)
    x, y_arr = promote_dtypes_inexact(x, y)
    if x.ndim != y_arr.ndim:
      raise ValueError("two-arguments must have the same rank ({x.ndim} vs {y.ndim}).")
    # Check if we can broadcast the outer axes together
    try:
      outershape = jnp.broadcast_shapes(tuple_delete(x.shape, axis),
                                        tuple_delete(y_arr.shape, axis))
    except ValueError as err:
      raise ValueError('x and y cannot be broadcast together.') from err

  result_dtype = dtypes.to_complex_dtype(x.dtype)
  freq_dtype = np.finfo(result_dtype).dtype

  nperseg_int: int = 0
  nfft_int: int = 0
  noverlap_int: int = 0

  if nperseg is not None:  # if specified by user
    nperseg_int = core.concrete_or_error(
        int, nperseg, "nperseg of windowed-FFT")
    if nperseg_int < 1:
      raise ValueError('nperseg must be a positive integer')
  # parse window; if array like, then set nperseg = win.shape
  win, nperseg_int = signal_helper._triage_segments(
      window, nperseg if nperseg is None else nperseg_int,
      input_length=x.shape[axis], dtype=x.dtype)

  if noverlap is None:
    noverlap_int = nperseg_int // 2
  else:
    noverlap_int = core.concrete_or_error(
        int, noverlap, "noverlap of windowed-FFT")

  if nfft is None:
    nfft_int = nperseg_int
  else:
    nfft_int = core.concrete_or_error(int, nfft, "nfft of windowed-FFT")

  # Special cases for size == 0
  if y is None:
    if x.size == 0:
      return jnp.zeros(x.shape, freq_dtype), jnp.zeros(x.shape, freq_dtype), jnp.zeros(x.shape, result_dtype)
  else:
    if x.size == 0 or y_arr.size == 0:
      shape = tuple_insert(outershape, min(x.shape[axis], y_arr.shape[axis]), axis)
      return jnp.zeros(shape, freq_dtype), jnp.zeros(shape, freq_dtype), jnp.zeros(shape, result_dtype)

  # Move time-axis to the end
  x = jnp.moveaxis(x, axis, -1)
  if y is not None and y_arr.ndim > 1:
    y_arr = jnp.moveaxis(y_arr, axis, -1)

  # Check if x and y are the same length, zero-pad if necessary
  if y is not None and x.shape[-1] != y_arr.shape[-1]:
    if x.shape[-1] < y_arr.shape[-1]:
      pad_shape = list(x.shape)
      pad_shape[-1] = y_arr.shape[-1] - x.shape[-1]
      x = jnp.concatenate((x, jnp.zeros_like(x, shape=pad_shape)), -1)
    else:
      pad_shape = list(y_arr.shape)
      pad_shape[-1] = x.shape[-1] - y_arr.shape[-1]
      y_arr = jnp.concatenate((y_arr, jnp.zeros_like(x, shape=pad_shape)), -1)

  if nfft_int < nperseg_int:
    raise ValueError('nfft must be greater than or equal to nperseg.')
  if noverlap_int >= nperseg_int:
    raise ValueError('noverlap must be less than nperseg.')
  nstep = nperseg_int - noverlap_int

  # Apply paddings
  if boundary is not None:
    ext_func = boundary_funcs[boundary]
    x = ext_func(x, nperseg_int // 2, axis=-1)
    if y is not None:
      y_arr = ext_func(y_arr, nperseg_int // 2, axis=-1)

  if padded:
    # Pad to integer number of windowed segments
    # I.e make x.shape[-1] = nperseg + (nseg-1)*nstep, with integer nseg
    nadd = (-(x.shape[-1]-nperseg_int) % nstep) % nperseg_int
    x = jnp.concatenate((x, jnp.zeros_like(x, shape=(*x.shape[:-1], nadd))), axis=-1)
    if y is not None:
      y_arr = jnp.concatenate((y_arr, jnp.zeros_like(x, shape=(*y_arr.shape[:-1], nadd))), axis=-1)

  # Handle detrending and window functions
  detrend_func: Any
  if isinstance(detrend_type, str):
    detrend_func = partial(detrend, type=detrend_type, axis=-1)
  elif callable(detrend_type):
    if axis != -1:
      # Wrap this function so that it receives a shape that it could
      # reasonably expect to receive.
      def detrend_func(d):
        d = jnp.moveaxis(d, axis, -1)
        d = detrend_type(d)
        return jnp.moveaxis(d, -1, axis)
    else:
      detrend_func = detrend_type
  elif not detrend_type:
    detrend_func = lambda d: d
  else:
    raise ValueError(f'Unsupported detrend type: {detrend_type}')

  # Determine scale
  if scaling == 'density':
    scale = 1.0 / (fs * (win * win).sum())
  elif scaling == 'spectrum':
    scale = 1.0 / win.sum()**2
  else:
    raise ValueError(f'Unknown scaling: {scaling}')
  if mode == 'stft':
    scale = jnp.sqrt(scale)
  scale, = promote_dtypes_complex(scale)

  # Determine onesided/ two-sided
  if return_onesided:
    sides = 'onesided'
    if jnp.iscomplexobj(x) or jnp.iscomplexobj(y):
      sides = 'twosided'
      warnings.warn('Input data is complex, switching to '
                    'return_onesided=False')
  else:
    sides = 'twosided'

  if sides == 'twosided':
    freqs = jax.numpy.fft.fftfreq(nfft_int, 1/fs, dtype=freq_dtype)
  elif sides == 'onesided':
    freqs = jax.numpy.fft.rfftfreq(nfft_int, 1/fs, dtype=freq_dtype)

  # Perform the windowed FFTs
  result = _fft_helper(x, win, detrend_func,
                       nperseg_int, noverlap_int, nfft_int, sides)

  if y is not None:
    # All the same operations on the y data
    result_y = _fft_helper(y_arr, win, detrend_func,
                           nperseg_int, noverlap_int, nfft_int, sides)
    result = jnp.conjugate(result) * result_y
  elif mode == 'psd':
    result = jnp.conjugate(result) * result

  result *= scale

  if sides == 'onesided' and mode == 'psd':
    end = None if nfft_int % 2 else -1
    result = result.at[..., 1:end].mul(2)

  time = jnp.arange(nperseg_int / 2, x.shape[-1] - nperseg_int / 2 + 1,
                    nperseg_int - noverlap_int, dtype=freq_dtype) / fs
  if boundary is not None:
    time -= (nperseg_int / 2) / fs

  result = result.astype(result_dtype)

  # All imaginary parts are zero anyways
  if y is None and mode != 'stft':
    result = result.real

  # Move frequency axis back to axis where the data came from
  result = jnp.moveaxis(result, -1, axis)

  return freqs, time, result


def stft(x: Array, fs: ArrayLike = 1.0, window: str = 'hann', nperseg: int = 256,
         noverlap: int | None = None, nfft: int | None = None,
         detrend: bool = False, return_onesided: bool = True, boundary: str | None = 'zeros',
         padded: bool = True, axis: int = -1) -> tuple[Array, Array, Array]:
  """
  Compute the short-time Fourier transform (STFT).

  JAX implementation of :func:`scipy.signal.stft`.

  Args:
    x: Array representing a time series of input values.
    fs: Sampling frequency of the time series (default: 1.0).
    window: Data tapering window to apply to each segment. Can be a window function name,
      a tuple specifying a window length and function, or an array (default: ``'hann'``).
    nperseg: Length of each segment (default: 256).
    noverlap: Number of points to overlap between segments (default: ``nperseg // 2``).
    nfft: Length of the FFT used, if a zero-padded FFT is desired. If ``None`` (default),
      the FFT length is ``nperseg``.
    detrend: Specifies how to detrend each segment. Can be ``False`` (default: no detrending),
      ``'constant'`` (remove mean), ``'linear'`` (remove linear trend), or a callable
      accepting a segment and returning a detrended segment.
    return_onesided: If True (default), return a one-sided spectrum for real inputs.
      If False, return a two-sided spectrum.
    boundary: Specifies whether the input signal is extended at both ends, and how.
      Options are ``None`` (no extension), ``'zeros'`` (default), ``'even'``, ``'odd'``,
      or ``'constant'``.
    padded: Specifies whether the input signal is zero-padded at the end to make its
      length a multiple of `nperseg`. If True (default), the padded signal length is
      the next multiple of ``nperseg``.
    axis: Axis along which the STFT is computed; the default is over the last axis (-1).

  Returns:
    A length-3 tuple of arrays ``(f, t, Zxx)``. ``f`` is the Array of sample frequencies.
    ``t`` is the Array of segment times, and ``Zxx`` is the STFT of ``x``.

  See Also:
    :func:`jax.scipy.signal.istft`: inverse short-time Fourier transform.
  """
  return _spectral_helper(x, None, fs, window, nperseg, noverlap,
                          nfft, detrend, return_onesided,
                          scaling='spectrum', axis=axis,
                          mode='stft', boundary=boundary,
                          padded=padded)


def csd(x: Array, y: ArrayLike | None, fs: ArrayLike = 1.0, window: str = 'hann',
        nperseg: int | None = None, noverlap: int | None = None,
        nfft: int | None = None, detrend: str = 'constant',
        return_onesided: bool = True, scaling: str = 'density',
        axis: int = -1, average: str = 'mean') -> tuple[Array, Array]:
  """
  Estimate cross power spectral density (CSD) using Welch's method.

  This is a JAX implementation of :func:`scipy.signal.csd`. It is similar to
  :func:`jax.scipy.signal.welch`, but it operates on two input signals and
  estimates their cross-spectral density instead of the power spectral density
  (PSD).

  Args:
    x: Array representing a time series of input values.
    y: Array representing the second time series of input values, the same length as ``x``
      along the specified ``axis``. If not specified, then assume ``y = x`` and compute
      the PSD ``Pxx`` of ``x`` via Welch's  method.
    fs: Sampling frequency of the inputs (default: 1.0).
    window: Data tapering window to apply to each segment. Can be a window function name,
      a tuple specifying a window length and function, or an array (default: ``'hann'``).
    nperseg: Length of each segment (default: 256).
    noverlap: Number of points to overlap between segments (default: ``nperseg // 2``).
    nfft: Length of the FFT used, if a zero-padded FFT is desired. If ``None`` (default),
      the FFT length is ``nperseg``.
    detrend: Specifies how to detrend each segment. Can be ``False`` (default: no detrending),
      ``'constant'`` (remove mean), ``'linear'`` (remove linear trend), or a callable
      accepting a segment and returning a detrended segment.
    return_onesided: If True (default), return a one-sided spectrum for real inputs.
      If False, return a two-sided spectrum.
    scaling: Selects between computing the power spectral density (``'density'``, default)
      or the power spectrum (``'spectrum'``)
    axis: Axis along which the CSD is computed (default: -1).
    average: The type of averaging to use on the periodograms; one of ``'mean'`` (default)
      or ``'median'``.

  Returns:
    A length-2 tuple of arrays ``(f, Pxy)``. ``f`` is the array of sample frequencies,
    and ``Pxy`` is the cross spectral density of `x` and `y`

  Notes:
    The original SciPy function exhibits slightly different behavior between
    ``csd(x, x)`` and ``csd(x, x.copy())``.  The LAX-backend version is designed
    to follow the latter behavior.  To replicate the former, call this function
    function as ``csd(x, None)``.

  See Also:
    - :func:`jax.scipy.signal.welch`: Power spectral density.
    - :func:`jax.scipy.signal.stft`: Short-time Fourier transform.
  """
  freqs, _, Pxy = _spectral_helper(x, y, fs, window, nperseg, noverlap, nfft,
                                  detrend, return_onesided, scaling, axis,
                                  mode='psd')
  if y is not None:
    Pxy = Pxy + 0j  # Ensure complex output when x is not y

  # Average over windows.
  if Pxy.ndim >= 2 and Pxy.size > 0:
    if Pxy.shape[-1] > 1:
      if average == 'median':
        bias = signal_helper._median_bias(Pxy.shape[-1]).astype(Pxy.dtype)
        if jnp.iscomplexobj(Pxy):
          Pxy = (jnp.median(jnp.real(Pxy), axis=-1)
                  + 1j * jnp.median(jnp.imag(Pxy), axis=-1))
        else:
          Pxy = jnp.median(Pxy, axis=-1)
        Pxy /= bias
      elif average == 'mean':
        Pxy = Pxy.mean(axis=-1)
      else:
        raise ValueError(f'average must be "median" or "mean", got {average}')
    else:
      Pxy = jnp.reshape(Pxy, Pxy.shape[:-1])

  return freqs, Pxy


def welch(x: Array, fs: ArrayLike = 1.0, window: str = 'hann',
          nperseg: int | None = None, noverlap: int | None = None,
          nfft: int | None = None, detrend: str = 'constant',
          return_onesided: bool = True, scaling: str = 'density',
          axis: int = -1, average: str = 'mean') -> tuple[Array, Array]:
  """
  Estimate power spectral density (PSD) using Welch's method.

  This is a JAX implementation of :func:`scipy.signal.welch`. It divides the
  input signal into overlapping segments, computes the modified periodogram for
  each segment, and averages the results to obtain a smoother estimate of the PSD.

  Args:
    x: Array representing a time series of input values.
    fs: Sampling frequency of the inputs (default: 1.0).
    window: Data tapering window to apply to each segment. Can be a window function name,
      a tuple specifying a window length and function, or an array (default: ``'hann'``).
    nperseg: Length of each segment (default: 256).
    noverlap: Number of points to overlap between segments (default: ``nperseg // 2``).
    nfft: Length of the FFT used, if a zero-padded FFT is desired. If ``None`` (default),
      the FFT length is ``nperseg``.
    detrend: Specifies how to detrend each segment. Can be ``False`` (default: no detrending),
      ``'constant'`` (remove mean), ``'linear'`` (remove linear trend), or a callable
      accepting a segment and returning a detrended segment.
    return_onesided: If True (default), return a one-sided spectrum for real inputs.
      If False, return a two-sided spectrum.
    scaling: Selects between computing the power spectral density (``'density'``, default)
      or the power spectrum (``'spectrum'``)
    axis: Axis along which the PSD is computed (default: -1).
    average: The type of averaging to use on the periodograms; one of ``'mean'`` (default)
      or ``'median'``.

  Returns:
    A length-2 tuple of arrays ``(f, Pxx)``. ``f`` is the array of sample frequencies,
    and ``Pxx`` is the power spectral density of ``x``.

  See Also:
    - :func:`jax.scipy.signal.csd`: Cross power spectral density.
    - :func:`jax.scipy.signal.stft`: Short-time Fourier transform.
  """
  freqs, Pxx = csd(x, None, fs=fs, window=window, nperseg=nperseg,
                   noverlap=noverlap, nfft=nfft, detrend=detrend,
                   return_onesided=return_onesided, scaling=scaling,
                   axis=axis, average=average)

  return freqs, Pxx.real


def _overlap_and_add(x: Array, step_size: int) -> Array:
  """Utility function compatible with tf.signal.overlap_and_add.

  Args:
    x: An array with `(..., frames, frame_length)`-shape.
    step_size: An integer denoting overlap offsets. Must be less than
      `frame_length`.

  Returns:
    An array with `(..., output_size)`-shape containing overlapped signal.
  """
  check_arraylike("_overlap_and_add", x)
  step_size = core.concrete_or_error(
      int, step_size, "step_size for overlap_and_add")
  if x.ndim < 2:
    raise ValueError('Input must have (..., frames, frame_length) shape.')

  *batch_shape, nframes, segment_len = x.shape
  flat_batchsize = math.prod(batch_shape)
  x = x.reshape((flat_batchsize, nframes, segment_len))
  output_size = step_size * (nframes - 1) + segment_len
  nstep_per_segment = 1 + (segment_len - 1) // step_size

  # Here, we use shorter notation for axes.
  # B: batch_size, N: nframes, S: nstep_per_segment,
  # T: segment_len divided by S

  padded_segment_len = nstep_per_segment * step_size
  x = jnp.pad(x, ((0, 0), (0, 0), (0, padded_segment_len - segment_len)))
  x = x.reshape((flat_batchsize, nframes, nstep_per_segment, step_size))

  # For obtaining shifted signals, this routine reinterprets flattened array
  # with a shrinked axis.  With appropriate truncation/ padding, this operation
  # pushes the last padded elements of the previous row to the head of the
  # current row.
  # See implementation of `overlap_and_add` in Tensorflow for details.
  x = x.transpose((0, 2, 1, 3))  # x: (B, S, N, T)
  x = jnp.pad(x, ((0, 0), (0, 0), (0, nframes), (0, 0)))  # x: (B, S, N*2, T)
  shrinked = x.shape[2] - 1
  x = x.reshape((flat_batchsize, -1))
  x = x[:, :(nstep_per_segment * shrinked * step_size)]
  x = x.reshape((flat_batchsize, nstep_per_segment, shrinked * step_size))

  # Finally, sum shifted segments, and truncate results to the output_size.
  x = x.sum(axis=1)[:, :output_size]
  return x.reshape(tuple(batch_shape) + (-1,))


def istft(Zxx: Array, fs: ArrayLike = 1.0, window: str = 'hann',
          nperseg: int | None = None, noverlap: int | None = None,
          nfft: int | None = None, input_onesided: bool = True,
          boundary: bool = True, time_axis: int = -1,
          freq_axis: int = -2) -> tuple[Array, Array]:
  """
  Perform the inverse short-time Fourier transform (ISTFT).

  JAX implementation of :func:`scipy.signal.istft`; computes the inverse of
  :func:`jax.scipy.signal.stft`.

  Args:
    Zxx: STFT of the signal to be reconstructed.
    fs: Sampling frequency of the time series (default: 1.0)
    window: Data tapering window to apply to each segment. Can be a window function name,
      a tuple specifying a window length and function, or an array (default: ``'hann'``).
    nperseg: Number of data points per segment in the STFT. If ``None`` (default), the
      value is determined from the size of ``Zxx``.
    noverlap: Number of points to overlap between segments (default: ``nperseg // 2``).
    nfft: Number of FFT points used in the STFT. If ``None`` (default), the
      value is determined from the size of ``Zxx``.
    input_onesided: If Tru` (default), interpret the input as a one-sided STFT
      (positive frequencies only). If False, interpret the input as a two-sided STFT.
    boundary: If True (default), it is assumed that the input signal was extended at
      its boundaries by ``stft``. If `False`, the input signal is assumed to have been truncated at the boundaries by `stft`.
    time_axis: Axis in `Zxx` corresponding to time segments (default: -1).
    freq_axis: Axis in `Zxx` corresponding to frequency bins (default: -2).

  Returns:
    A length-2 tuple of arrays ``(t, x)``. ``t`` is the Array of signal times, and ``x``
    is the reconstructed time series.

  See Also:
    :func:`jax.scipy.signal.stft`: short-time Fourier transform.

  Examples:
    Demonstrate that this gives the inverse of :func:`~jax.scipy.signal.stft`:

    >>> x = jnp.array([1., 2., 3., 2., 1., 0., 1., 2.])
    >>> f, t, Zxx = jax.scipy.signal.stft(x, nperseg=4)
    >>> print(Zxx)  # doctest: +SKIP
    [[ 1. +0.j   2.5+0.j   1. +0.j   1. +0.j   0.5+0.j ]
     [-0.5+0.5j -1.5+0.j  -0.5-0.5j -0.5+0.5j  0. -0.5j]
     [ 0. +0.j   0.5+0.j   0. +0.j   0. +0.j  -0.5+0.j ]]
    >>> t, x_reconstructed = jax.scipy.signal.istft(Zxx)
    >>> print(x_reconstructed)
    [1. 2. 3. 2. 1. 0. 1. 2.]
  """
  # Input validation
  check_arraylike("istft", Zxx)
  if Zxx.ndim < 2:
    raise ValueError('Input stft must be at least 2d!')
  freq_axis = canonicalize_axis(freq_axis, Zxx.ndim)
  time_axis = canonicalize_axis(time_axis, Zxx.ndim)
  if freq_axis == time_axis:
    raise ValueError('Must specify differing time and frequency axes!')

  Zxx = jnp.asarray(Zxx, dtype=jax.dtypes.canonicalize_dtype(
      np.result_type(Zxx, np.complex64)))

  n_default = (2 * (Zxx.shape[freq_axis] - 1) if input_onesided
               else Zxx.shape[freq_axis])

  nperseg_int = core.concrete_or_error(int, nperseg or n_default,
                                           "nperseg: segment length of STFT")
  if nperseg_int < 1:
    raise ValueError('nperseg must be a positive integer')

  nfft_int: int = 0
  if nfft is None:
    nfft_int = n_default
    if input_onesided and nperseg_int == n_default + 1:
      nfft_int += 1  # Odd nperseg, no FFT padding
  else:
    nfft_int = core.concrete_or_error(int, nfft, "nfft of STFT")
  if nfft_int < nperseg_int:
    raise ValueError(
        f'FFT length ({nfft_int}) must be longer than nperseg ({nperseg_int}).')

  noverlap_int = core.concrete_or_error(
      int, noverlap or nperseg_int // 2, "noverlap of STFT")
  if noverlap_int >= nperseg_int:
    raise ValueError('noverlap must be less than nperseg.')
  nstep = nperseg_int - noverlap_int

  # Rearrange axes if necessary
  if time_axis != Zxx.ndim - 1 or freq_axis != Zxx.ndim - 2:
    outer_idxs = tuple(
        idx for idx in range(Zxx.ndim) if idx not in {time_axis, freq_axis})
    Zxx = jnp.transpose(Zxx, outer_idxs + (freq_axis, time_axis))

  # Perform IFFT
  ifunc = jax.numpy.fft.irfft if input_onesided else jax.numpy.fft.ifft
  # xsubs: [..., T, N], N is the number of frames, T is the frame length.
  xsubs = ifunc(Zxx, axis=-2, n=nfft)[..., :nperseg_int, :]

  # Get window as array
  if window == 'hann':
    # Implement the default case without scipy
    win = jnp.array([1.0]) if nperseg_int == 1 else jnp.sin(jnp.linspace(0, jnp.pi, nperseg_int, endpoint=False)) ** 2
    win = win.astype(xsubs.dtype)
  elif isinstance(window, (str, tuple)):
    # TODO(jakevdp): implement get_window() in JAX to remove optional scipy dependency
    try:
      from scipy.signal import get_window
    except ImportError as err:
      raise ImportError(f"scipy must be available to use {window=}") from err
    win = get_window(window, nperseg_int)
    win = jnp.array(win, dtype=xsubs.dtype)
  else:
    win = jnp.asarray(window)
    if len(win.shape) != 1:
      raise ValueError('window must be 1-D')
    if win.shape[0] != nperseg_int:
      raise ValueError(f'window must have length of {nperseg_int}')
  xsubs *= win.sum()  # This takes care of the 'spectrum' scaling

  # make win broadcastable over xsubs
  win = lax.expand_dims(win, (*range(xsubs.ndim - 2), -1))
  x = _overlap_and_add((xsubs * win).swapaxes(-2, -1), nstep)
  win_squared = jnp.repeat((win * win), xsubs.shape[-1], axis=-1)
  norm = _overlap_and_add(win_squared.swapaxes(-2, -1), nstep)

  # Remove extension points
  if boundary:
    x = x[..., nperseg_int//2:-(nperseg_int//2)]
    norm = norm[..., nperseg_int//2:-(nperseg_int//2)]
  x /= jnp.where(norm > 1e-10, norm, 1.0)

  # Put axes back
  if x.ndim > 1:
    if time_axis != Zxx.ndim - 1:
      if freq_axis < time_axis:
        time_axis -= 1
      x = jnp.moveaxis(x, -1, time_axis)

  time = jnp.arange(x.shape[0], dtype=np.finfo(x.dtype).dtype) / fs
  return time, x
