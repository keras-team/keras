# Copyright 2021 The JAX Authors.
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

from collections.abc import Sequence
from functools import partial
import math

from jax import lax
import jax.numpy as jnp
from jax._src.util import canonicalize_axis
from jax._src.numpy.util import promote_dtypes_complex, promote_dtypes_inexact
from jax._src.typing import Array

def _W4(N: int, k: Array) -> Array:
  N_arr, k = promote_dtypes_complex(N, k)
  return jnp.exp(-.5j * jnp.pi * k / N_arr)

def _dct_interleave(x: Array, axis: int) -> Array:
  v0 = lax.slice_in_dim(x, None, None, 2, axis)
  v1 = lax.rev(lax.slice_in_dim(x, 1, None, 2, axis), (axis,))
  return lax.concatenate([v0, v1], axis)

def _dct_ortho_norm(out: Array, axis: int) -> Array:
  factor = lax.concatenate([lax.full((1,), 4, out.dtype), lax.full((out.shape[axis] - 1,), 2, out.dtype)], 0)
  factor = lax.expand_dims(factor, [a for a in range(out.ndim) if a != axis])
  return out / lax.sqrt(factor * out.shape[axis])

# Implementation based on
# John Makhoul: A Fast Cosine Transform in One and Two Dimensions (1980)


def dct(x: Array, type: int = 2, n: int | None = None,
        axis: int = -1, norm: str | None = None) -> Array:
  """Computes the discrete cosine transform of the input

  JAX implementation of :func:`scipy.fft.dct`.

  Args:
    x: array
    type: integer, default = 2. Currently only type 2 is supported.
    n: integer, default = x.shape[axis]. The length of the transform.
      If larger than ``x.shape[axis]``, the input will be zero-padded, if
      smaller, the input will be truncated.
    axis: integer, default=-1. The axis along which the dct will be performed.
    norm: string. The normalization mode: one of ``[None, "backward", "ortho"]``.
      The default is ``None``, which is equivalent to ``"backward"``.

  Returns:
    array containing the discrete cosine transform of x

  See Also:
    - :func:`jax.scipy.fft.dctn`: multidimensional DCT
    - :func:`jax.scipy.fft.idct`: inverse DCT
    - :func:`jax.scipy.fft.idctn`: multidimensional inverse DCT

  Examples:
    >>> x = jax.random.normal(jax.random.key(0), (3, 3))
    >>> with jnp.printoptions(precision=2, suppress=True):
    ...   print(jax.scipy.fft.dct(x))
    [[ 6.43  3.56 -2.86]
     [-1.75  1.55 -1.4 ]
     [ 1.33 -2.01 -0.82]]

    When ``n`` smaller than ``x.shape[axis]``

    >>> with jnp.printoptions(precision=2, suppress=True):
    ...   print(jax.scipy.fft.dct(x, n=2))
    [[ 7.3  -0.57]
     [ 0.19 -0.36]
     [-0.   -1.4 ]]

    When ``n`` smaller than ``x.shape[axis]`` and ``axis=0``

    >>> with jnp.printoptions(precision=2, suppress=True):
    ...   print(jax.scipy.fft.dct(x, n=2, axis=0))
    [[ 3.09  4.4  -2.81]
     [ 2.41  2.62  0.76]]

    When ``n`` larger than ``x.shape[axis]`` and ``axis=1``

    >>> with jnp.printoptions(precision=2, suppress=True):
    ...   print(jax.scipy.fft.dct(x, n=4, axis=1))
    [[ 6.43  4.88  0.04 -3.3 ]
     [-1.75  0.73  1.01 -2.18]
     [ 1.33 -1.05 -2.34 -0.07]]
  """
  if type != 2:
    raise NotImplementedError('Only DCT type 2 is implemented.')
  if norm is not None and norm not in ['backward', 'ortho']:
    raise ValueError(f"jax.scipy.fft.dct: {norm=!r} is not implemented")

  axis = canonicalize_axis(axis, x.ndim)
  if n is not None:
    x = lax.pad(x, jnp.array(0, x.dtype),
                [(0, n - x.shape[axis] if a == axis else 0, 0)
                 for a in range(x.ndim)])

  N = x.shape[axis]
  v = _dct_interleave(x, axis)
  V = jnp.fft.fft(v, axis=axis)
  k = lax.expand_dims(jnp.arange(N, dtype=V.real.dtype), [a for a in range(x.ndim) if a != axis])
  out = V * _W4(N, k)
  out = 2 * out.real
  if norm == 'ortho':
    out = _dct_ortho_norm(out, axis)
  return out


def _dct2(x: Array, axes: Sequence[int], norm: str | None) -> Array:
  axis1, axis2 = map(partial(canonicalize_axis, num_dims=x.ndim), axes)
  N1, N2 = x.shape[axis1], x.shape[axis2]
  v = _dct_interleave(_dct_interleave(x, axis1), axis2)
  V = jnp.fft.fftn(v, axes=axes)
  k1 = lax.expand_dims(jnp.arange(N1, dtype=V.dtype),
                       [a for a in range(x.ndim) if a != axis1])
  k2 = lax.expand_dims(jnp.arange(N2, dtype=V.dtype),
                       [a for a in range(x.ndim) if a != axis2])
  out = _W4(N1, k1) * (_W4(N2, k2) * V + _W4(N2, -k2) * jnp.roll(jnp.flip(V, axis=axis2), shift=1, axis=axis2))
  out = 2 * out.real
  if norm == 'ortho':
    return _dct_ortho_norm(_dct_ortho_norm(out, axis1), axis2)
  return out


def dctn(x: Array, type: int = 2,
         s: Sequence[int] | None=None,
         axes: Sequence[int] | None = None,
         norm: str | None = None) -> Array:
  """Computes the multidimensional discrete cosine transform of the input

  JAX implementation of :func:`scipy.fft.dctn`.

  Args:
    x: array
    type: integer, default = 2. Currently only type 2 is supported.
    s: integer or sequence of integers. Specifies the shape of the result. If not
      specified, it will default to the shape of ``x`` along the specified ``axes``.
    axes: integer or sequence of integers. Specifies the axes along which the
      transform will be computed.
    norm: string. The normalization mode: one of ``[None, "backward", "ortho"]``.
      The default is ``None``, which is equivalent to ``"backward"``.

  Returns:
    array containing the discrete cosine transform of x

  See Also:
    - :func:`jax.scipy.fft.dct`: one-dimensional DCT
    - :func:`jax.scipy.fft.idct`: one-dimensional inverse DCT
    - :func:`jax.scipy.fft.idctn`: multidimensional inverse DCT

  Examples:

    ``jax.scipy.fft.dctn`` computes the transform along both the axes by default
    when ``axes`` argument is ``None``.

    >>> x = jax.random.normal(jax.random.key(0), (3, 3))
    >>> with jnp.printoptions(precision=2, suppress=True):
    ...   print(jax.scipy.fft.dctn(x))
    [[ 12.01   6.2  -10.17]
     [  8.84   9.65  -3.54]
     [ 11.25  -1.54  -0.88]]

    When ``s=[2]``, dimension of the transform along ``axis 0`` will be ``2``
    and dimension along ``axis 1`` will be same as that of input.

    >>> with jnp.printoptions(precision=2, suppress=True):
    ...   print(jax.scipy.fft.dctn(x, s=[2]))
    [[ 9.36 10.22 -8.53]
     [11.57  2.85 -2.06]]

    When ``s=[2]`` and ``axes=[1]``, dimension of the transform along ``axis 1`` will
    be ``2`` and dimension along ``axis 0`` will  be same as that of input.
    Also when ``axes=[1]``, transform will be computed only along ``axis 1``.

    >>> with jnp.printoptions(precision=2, suppress=True):
    ...   print(jax.scipy.fft.dctn(x, s=[2], axes=[1]))
    [[ 7.3  -0.57]
     [ 0.19 -0.36]
     [-0.   -1.4 ]]

    When ``s=[2, 4]``, shape of the transform will be ``(2, 4)``.

    >>> with jnp.printoptions(precision=2, suppress=True):
    ...   print(jax.scipy.fft.dctn(x, s=[2, 4]))
    [[  9.36  11.23   2.12 -10.97]
     [ 11.57   5.86  -1.37  -1.58]]
"""
  if type != 2:
    raise NotImplementedError('Only DCT type 2 is implemented.')
  if norm is not None and norm not in ['backward', 'ortho']:
    raise ValueError(f"jax.scipy.fft.dctn: {norm=!r} is not implemented")

  if axes is None:
    axes = range(x.ndim)

  if len(axes) == 1:
    return dct(x, n=s[0] if s is not None else None, axis=axes[0], norm=norm)

  if s is not None:
    ns = dict(zip(axes, s))
    pads = [(0, ns[a] - x.shape[a] if a in ns else 0, 0) for a in range(x.ndim)]
    x = lax.pad(x, jnp.array(0, x.dtype), pads)

  if len(axes) == 2:
    return _dct2(x, axes=axes, norm=norm)

  # compose high-D DCTs from 2D and 1D DCTs:
  for axes_block in [axes[i:i+2] for i in range(0, len(axes), 2)]:
    x = dctn(x, axes=axes_block, norm=norm)
  return x


def idct(x: Array, type: int = 2, n: int | None = None,
        axis: int = -1, norm: str | None = None) -> Array:
  """Computes the inverse discrete cosine transform of the input

  JAX implementation of :func:`scipy.fft.idct`.

  Args:
    x: array
    type: integer, default = 2. Currently only type 2 is supported.
    n: integer, default = x.shape[axis]. The length of the transform.
      If larger than ``x.shape[axis]``, the input will be zero-padded, if
      smaller, the input will be truncated.
    axis: integer, default=-1. The axis along which the dct will be performed.
    norm: string. The normalization mode: one of ``[None, "backward", "ortho"]``.
      The default is ``None``, which is equivalent to ``"backward"``.

  Returns:
    array containing the inverse discrete cosine transform of x

  See Also:
    - :func:`jax.scipy.fft.dct`: DCT
    - :func:`jax.scipy.fft.dctn`: multidimensional DCT
    - :func:`jax.scipy.fft.idctn`: multidimensional inverse DCT

  Examples:

    >>> x = jax.random.normal(jax.random.key(0), (3, 3))
    >>> with jnp.printoptions(precision=2, suppress=True):
    ...    print(jax.scipy.fft.idct(x))
    [[ 0.78  0.41 -0.39]
     [-0.12  0.31 -0.23]
     [ 0.17 -0.3  -0.11]]

    When ``n`` smaller than ``x.shape[axis]``

    >>> with jnp.printoptions(precision=2, suppress=True):
    ...    print(jax.scipy.fft.idct(x, n=2))
    [[ 1.12 -0.31]
     [ 0.04 -0.08]
     [ 0.05 -0.3 ]]

    When ``n`` smaller than ``x.shape[axis]`` and ``axis=0``

    >>> with jnp.printoptions(precision=2, suppress=True):
    ...    print(jax.scipy.fft.idct(x, n=2, axis=0))
    [[ 0.38  0.57 -0.45]
     [ 0.43  0.44  0.24]]

    When ``n`` larger than ``x.shape[axis]`` and ``axis=0``

    >>> with jnp.printoptions(precision=2, suppress=True):
    ...    print(jax.scipy.fft.idct(x, n=4, axis=0))
    [[ 0.1   0.38 -0.16]
     [ 0.28  0.18 -0.26]
     [ 0.3   0.15 -0.08]
     [ 0.13  0.3   0.29]]

    ``jax.scipy.fft.idct`` can be used to reconstruct ``x`` from the result
    of ``jax.scipy.fft.dct``

    >>> x_dct = jax.scipy.fft.dct(x)
    >>> jnp.allclose(x, jax.scipy.fft.idct(x_dct))
    Array(True, dtype=bool)
  """
  if type != 2:
    raise NotImplementedError('Only DCT type 2 is implemented.')
  if norm is not None and norm not in ['backward', 'ortho']:
    raise ValueError(f"jax.scipy.fft.idct: {norm=!r} is not implemented")

  axis = canonicalize_axis(axis, x.ndim)
  if n is not None:
    x = lax.pad(x, jnp.array(0, x.dtype),
                [(0, n - x.shape[axis] if a == axis else 0, 0)
                 for a in range(x.ndim)])
  N = x.shape[axis]
  x, = promote_dtypes_inexact(x)
  if norm is None or norm == 'backward':
    x = _dct_ortho_norm(x, axis)
  x = _dct_ortho_norm(x, axis)

  k = lax.expand_dims(jnp.arange(N, dtype=x.dtype), [a for a in range(x.ndim) if a != axis])
  # everything is complex from here...
  w4 = _W4(N,k)
  x = x.astype(w4.dtype)
  x = x / (_W4(N, k))
  x = x * 2 * N

  x = jnp.fft.ifft(x, axis=axis)
  # convert back to reals..
  out = _dct_deinterleave(x.real, axis)
  return out


def idctn(x: Array, type: int = 2,
          s: Sequence[int] | None=None,
          axes: Sequence[int] | None = None,
          norm: str | None = None) -> Array:
  """Computes the multidimensional inverse discrete cosine transform of the input

  JAX implementation of :func:`scipy.fft.idctn`.

  Args:
    x: array
    type: integer, default = 2. Currently only type 2 is supported.
    s: integer or sequence of integers. Specifies the shape of the result. If not
      specified, it will default to the shape of ``x`` along the specified ``axes``.
    axes: integer or sequence of integers. Specifies the axes along which the
      transform will be computed.
    norm: string. The normalization mode: one of ``[None, "backward", "ortho"]``.
      The default is ``None``, which is equivalent to ``"backward"``.

  Returns:
    array containing the inverse discrete cosine transform of x

  See Also:
    - :func:`jax.scipy.fft.dct`: one-dimensional DCT
    - :func:`jax.scipy.fft.dctn`: multidimensional DCT
    - :func:`jax.scipy.fft.idct`: one-dimensional inverse DCT

  Examples:

    ``jax.scipy.fft.idctn`` computes the transform along both the axes by default
    when ``axes`` argument is ``None``.

    >>> x = jax.random.normal(jax.random.key(0), (3, 3))
    >>> with jnp.printoptions(precision=2, suppress=True):
    ...    print(jax.scipy.fft.idctn(x))
    [[ 0.12  0.11 -0.15]
     [ 0.07  0.17 -0.03]
     [ 0.19 -0.07 -0.02]]

    When ``s=[2]``, dimension of the transform along ``axis 0`` will be ``2``
    and dimension along ``axis 1`` will be the same as that of input.

    >>> with jnp.printoptions(precision=2, suppress=True):
    ...  print(jax.scipy.fft.idctn(x, s=[2]))
    [[ 0.15  0.21 -0.18]
     [ 0.24 -0.01 -0.02]]

    When ``s=[2]`` and ``axes=[1]``, dimension of the transform along ``axis 1`` will
    be ``2`` and dimension along ``axis 0`` will  be same as that of input.
    Also when ``axes=[1]``, transform will be computed only along ``axis 1``.

    >>> with jnp.printoptions(precision=2, suppress=True):
    ...  print(jax.scipy.fft.idctn(x, s=[2], axes=[1]))
    [[ 1.12 -0.31]
     [ 0.04 -0.08]
     [ 0.05 -0.3 ]]

    When ``s=[2, 4]``, shape of the transform will be ``(2, 4)``

    >>> with jnp.printoptions(precision=2, suppress=True):
    ...  print(jax.scipy.fft.idctn(x, s=[2, 4]))
    [[ 0.1   0.18  0.07 -0.16]
     [ 0.2   0.06 -0.03 -0.01]]

    ``jax.scipy.fft.idctn`` can be used to reconstruct ``x`` from the result
    of ``jax.scipy.fft.dctn``

    >>> x_dctn = jax.scipy.fft.dctn(x)
    >>> jnp.allclose(x, jax.scipy.fft.idctn(x_dctn))
    Array(True, dtype=bool)
  """
  if type != 2:
    raise NotImplementedError('Only DCT type 2 is implemented.')
  if norm is not None and norm not in ['backward', 'ortho']:
    raise ValueError(f"jax.scipy.fft.idctn: {norm=!r} is not implemented")

  if axes is None:
    axes = range(x.ndim)

  if len(axes) == 1:
    return idct(x, n=s[0] if s is not None else None, axis=axes[0], norm=norm)

  if s is not None:
    ns = dict(zip(axes, s))
    pads = [(0, ns[a] - x.shape[a] if a in ns else 0, 0) for a in range(x.ndim)]
    x = lax.pad(x, jnp.array(0, x.dtype), pads)

  # compose high-D DCTs from 1D DCTs:
  for axis in axes:
    x = idct(x, axis=axis, norm=norm)
  return x


def _dct_deinterleave(x: Array, axis: int) -> Array:
  empty_slice = slice(None, None, None)
  ix0 = tuple(
      slice(None, math.ceil(x.shape[axis]/2), 1) if i == axis else empty_slice
      for i in range(len(x.shape)))
  ix1  = tuple(
      slice(math.ceil(x.shape[axis]/2), None, 1) if i == axis else empty_slice
      for i in range(len(x.shape)))
  v0 = x[ix0]
  v1 = lax.rev(x[ix1], (axis,))
  out = jnp.zeros(x.shape, dtype=x.dtype)
  evens = tuple(
      slice(None, None, 2) if i == axis else empty_slice for i in range(len(x.shape)))
  odds = tuple(
      slice(1, None, 2) if i == axis else empty_slice for i in range(len(x.shape)))
  out =  out.at[evens].set(v0)
  out = out.at[odds].set(v1)
  return out
