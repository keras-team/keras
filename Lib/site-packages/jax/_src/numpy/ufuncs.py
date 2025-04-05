# Copyright 2018 The JAX Authors.
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

"""
Implements ufuncs for jax.numpy.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import partial
import operator
from typing import Any

import numpy as np

from jax._src import core
from jax._src import dtypes
from jax._src.api import jit
from jax._src.custom_derivatives import custom_jvp
from jax._src.lax import lax
from jax._src.lax import other as lax_other
from jax._src.typing import Array, ArrayLike
from jax._src.numpy.util import (
   check_arraylike, promote_args, promote_args_inexact,
   promote_args_numeric, promote_dtypes_inexact, promote_dtypes_numeric,
   promote_shapes, _where, check_no_float0s)
from jax._src.numpy.ufunc_api import ufunc
from jax._src.numpy import reductions
from jax._src.util import set_module


export = set_module('jax.numpy')

_lax_const = lax._const

_INT_DTYPES = {
  16: np.int16,
  32: np.int32,
  64: np.int64,
}

def _constant_like(x, const):
  return np.array(const, dtype=dtypes.dtype(x))

def _replace_inf(x: ArrayLike) -> Array:
  return lax.select(isposinf(real(x)), lax._zeros(x), x)

def _to_bool(x: Array) -> Array:
  return x if x.dtype == bool else lax.ne(x, _lax_const(x, 0))


def unary_ufunc(func: Callable[[ArrayLike], Array]) -> ufunc:
  """An internal helper function for defining unary ufuncs."""
  func_jit = jit(func, inline=True)
  return ufunc(func_jit, name=func.__name__, nin=1, nout=1, call=func_jit)


def binary_ufunc(identity: Any, reduce: Callable[..., Any] | None = None,
                 accumulate: Callable[..., Any] | None = None,
                 at: Callable[..., Any] | None = None,
                 reduceat: Callable[..., Any] | None = None) -> Callable[[Callable[[ArrayLike, ArrayLike], Array]], ufunc]:
  """An internal helper function for defining binary ufuncs."""
  def decorator(func: Callable[[ArrayLike, ArrayLike], Array]) -> ufunc:
    func_jit = jit(func, inline=True)
    return ufunc(func_jit, name=func.__name__, nin=2, nout=1, call=func_jit,
                 identity=identity, reduce=reduce, accumulate=accumulate, at=at, reduceat=reduceat)
  return decorator


@export
@partial(jit, inline=True)
def fabs(x: ArrayLike, /) -> Array:
  """Compute the element-wise absolute values of the real-valued input.

  JAX implementation of :obj:`numpy.fabs`.

  Args:
    x: input array or scalar. Must not have a complex dtype.

  Returns:
    An array with same shape as ``x`` and dtype float, containing the element-wise
    absolute values.

  See also:
    - :func:`jax.numpy.absolute`: Computes the absolute values of the input including
      complex dtypes.
    - :func:`jax.numpy.abs`: Computes the absolute values of the input including
      complex dtypes.

  Examples:
    For integer inputs:

    >>> x = jnp.array([-5, -9, 1, 10, 15])
    >>> jnp.fabs(x)
    Array([ 5.,  9.,  1., 10., 15.], dtype=float32)

    For float type inputs:

    >>> x1 = jnp.array([-1.342, 5.649, 3.927])
    >>> jnp.fabs(x1)
    Array([1.342, 5.649, 3.927], dtype=float32)

    For boolean inputs:

    >>> x2 = jnp.array([True, False])
    >>> jnp.fabs(x2)
    Array([1., 0.], dtype=float32)
  """
  check_arraylike('fabs', x)
  if dtypes.issubdtype(dtypes.dtype(x), np.complexfloating):
    raise TypeError("ufunc 'fabs' does not support complex dtypes")
  return lax.abs(*promote_args_inexact('fabs', x))


@export
@partial(jit, inline=True)
def bitwise_invert(x: ArrayLike, /) -> Array:
  """Alias of :func:`jax.numpy.invert`."""
  return lax.bitwise_not(*promote_args('bitwise_invert', x))


@export
@partial(jit, inline=True)
def bitwise_not(x: ArrayLike, /) -> Array:
  """Alias of :func:`jax.numpy.invert`."""
  return lax.bitwise_not(*promote_args('bitwise_not', x))


@export
@partial(jit, inline=True)
def invert(x: ArrayLike, /) -> Array:
  """Compute the bitwise inversion of an input.

  JAX implementation of :func:`numpy.invert`. This function provides the
  implementation of the ``~`` operator for JAX arrays.

  Args:
    x: input array, must be boolean or integer typed.

  Returns:
    An array of the same shape and dtype as ```x``, with the bits inverted.

  See also:
    - :func:`jax.numpy.bitwise_invert`: Array API alias of this function.
    - :func:`jax.numpy.logical_not`: Invert after casting input to boolean.

  Examples:
    >>> x = jnp.arange(5, dtype='uint8')
    >>> print(x)
    [0 1 2 3 4]
    >>> print(jnp.invert(x))
    [255 254 253 252 251]

    This function implements the unary ``~`` operator for JAX arrays:

    >>> print(~x)
    [255 254 253 252 251]

    :func:`invert` operates bitwise on the input, and so the meaning of its
    output may be more clear by showing the bitwise representation:

    >>> with jnp.printoptions(formatter={'int': lambda x: format(x, '#010b')}):
    ...   print(f"{x  = }")
    ...   print(f"{~x = }")
    x  = Array([0b00000000, 0b00000001, 0b00000010, 0b00000011, 0b00000100], dtype=uint8)
    ~x = Array([0b11111111, 0b11111110, 0b11111101, 0b11111100, 0b11111011], dtype=uint8)

    For boolean inputs, :func:`invert` is equivalent to :func:`logical_not`:

    >>> x = jnp.array([True, False, True, True, False])
    >>> jnp.invert(x)
    Array([False,  True, False, False,  True], dtype=bool)
  """
  return lax.bitwise_not(*promote_args('invert', x))


@unary_ufunc
def negative(x: ArrayLike, /) -> Array:
  """Return element-wise negative values of the input.

  JAX implementation of :obj:`numpy.negative`.

  Args:
    x: input array or scalar.

  Returns:
    An array with same shape and dtype as ``x`` containing ``-x``.

  See also:
    - :func:`jax.numpy.positive`: Returns element-wise positive values of the input.
    - :func:`jax.numpy.sign`: Returns element-wise indication of sign of the input.

  Note:
    ``jnp.negative``, when applied over ``unsigned integer``, produces the result
    of their two's complement negation, which typically results in unexpected
    large positive values due to integer underflow.

  Examples:
    For real-valued inputs:

    >>> x = jnp.array([0., -3., 7])
    >>> jnp.negative(x)
    Array([-0.,  3., -7.], dtype=float32)

    For complex inputs:

    >>> x1 = jnp.array([1-2j, -3+4j, 5-6j])
    >>> jnp.negative(x1)
    Array([-1.+2.j,  3.-4.j, -5.+6.j], dtype=complex64)

    For unit32:

    >>> x2 = jnp.array([5, 0, -7]).astype(jnp.uint32)
    >>> x2
    Array([         5,          0, 4294967289], dtype=uint32)
    >>> jnp.negative(x2)
    Array([4294967291,          0,          7], dtype=uint32)
  """
  return lax.neg(*promote_args('negative', x))


@export
@partial(jit, inline=True)
def positive(x: ArrayLike, /) -> Array:
  """Return element-wise positive values of the input.

  JAX implementation of :obj:`numpy.positive`.

  Args:
    x: input array or scalar

  Returns:
    An array of same shape and dtype as ``x`` containing ``+x``.

  Note:
    ``jnp.positive`` is equivalent to ``x.copy()`` and is defined only for the
    types that support arithmetic operations.

  See also:
    - :func:`jax.numpy.negative`: Returns element-wise negative values of the input.
    - :func:`jax.numpy.sign`: Returns element-wise indication of sign of the input.

  Examples:
    For real-valued inputs:

    >>> x = jnp.array([-5, 4, 7., -9.5])
    >>> jnp.positive(x)
    Array([-5. ,  4. ,  7. , -9.5], dtype=float32)
    >>> x.copy()
    Array([-5. ,  4. ,  7. , -9.5], dtype=float32)

    For complex inputs:

    >>> x1 = jnp.array([1-2j, -3+4j, 5-6j])
    >>> jnp.positive(x1)
    Array([ 1.-2.j, -3.+4.j,  5.-6.j], dtype=complex64)
    >>> x1.copy()
    Array([ 1.-2.j, -3.+4.j,  5.-6.j], dtype=complex64)

    For uint32:

    >>> x2 = jnp.array([6, 0, -4]).astype(jnp.uint32)
    >>> x2
    Array([         6,          0, 4294967292], dtype=uint32)
    >>> jnp.positive(x2)
    Array([         6,          0, 4294967292], dtype=uint32)
  """
  return lax.asarray(*promote_args('positive', x))


@export
@partial(jit, inline=True)
def sign(x: ArrayLike, /) -> Array:
  r"""Return an element-wise indication of sign of the input.

  JAX implementation of :obj:`numpy.sign`.

  The sign of ``x`` for real-valued input is:

  .. math::
    \mathrm{sign}(x) = \begin{cases}
      1, & x > 0\\
      0, & x = 0\\
      -1, & x < 0
    \end{cases}

  For complex valued input, ``jnp.sign`` returns a unit vector repesenting the
  phase. For generalized case, the sign of ``x`` is given by:

  .. math::
    \mathrm{sign}(x) = \begin{cases}
      \frac{x}{abs(x)}, & x \ne 0\\
      0, & x = 0
    \end{cases}

  Args:
    x: input array or scalar.

  Returns:
    An array with same shape and dtype as ``x`` containing the sign indication.

  See also:
    - :func:`jax.numpy.positive`: Returns element-wise positive values of the input.
    - :func:`jax.numpy.negative`: Returns element-wise negative values of the input.

  Examples:
    For Real-valued inputs:

    >>> x = jnp.array([0., -3., 7.])
    >>> jnp.sign(x)
    Array([ 0., -1.,  1.], dtype=float32)

    For complex-inputs:

    >>> x1 = jnp.array([1, 3+4j, 5j])
    >>> jnp.sign(x1)
    Array([1. +0.j , 0.6+0.8j, 0. +1.j ], dtype=complex64)
  """
  return lax.sign(*promote_args('sign', x))


@export
@partial(jit, inline=True)
def floor(x: ArrayLike, /) -> Array:
  """Round input to the nearest integer downwards.

  JAX implementation of :obj:`numpy.floor`.

  Args:
    x: input array or scalar. Must not have complex dtype.

  Returns:
    An array with same shape and dtype as ``x`` containing the values rounded to
    the nearest integer that is less than or equal to the value itself.

  See also:
    - :func:`jax.numpy.fix`: Rounds the input to the nearest interger towards zero.
    - :func:`jax.numpy.trunc`: Rounds the input to the nearest interger towards
      zero.
    - :func:`jax.numpy.ceil`: Rounds the input up to the nearest integer.

  Examples:
    >>> key = jax.random.key(42)
    >>> x = jax.random.uniform(key, (3, 3), minval=-5, maxval=5)
    >>> with jnp.printoptions(precision=2, suppress=True):
    ...     print(x)
    [[-0.11  1.8   1.16]
     [ 0.61 -0.49  0.86]
     [-4.25  2.75  1.99]]
    >>> jnp.floor(x)
    Array([[-1.,  1.,  1.],
           [ 0., -1.,  0.],
           [-5.,  2.,  1.]], dtype=float32)
  """
  check_arraylike('floor', x)
  if dtypes.isdtype(dtypes.dtype(x), ('integral', 'bool')):
    return lax.asarray(x)
  return lax.floor(*promote_args_inexact('floor', x))


@export
@partial(jit, inline=True)
def ceil(x: ArrayLike, /) -> Array:
  """Round input to the nearest integer upwards.

  JAX implementation of :obj:`numpy.ceil`.

  Args:
    x: input array or scalar. Must not have complex dtype.

  Returns:
    An array with same shape and dtype as ``x`` containing the values rounded to
    the nearest integer that is greater than or equal to the value itself.

  See also:
    - :func:`jax.numpy.fix`: Rounds the input to the nearest interger towards zero.
    - :func:`jax.numpy.trunc`: Rounds the input to the nearest interger towards
      zero.
    - :func:`jax.numpy.floor`: Rounds the input down to the nearest integer.

  Examples:
    >>> key = jax.random.key(1)
    >>> x = jax.random.uniform(key, (3, 3), minval=-5, maxval=5)
    >>> with jnp.printoptions(precision=2, suppress=True):
    ...     print(x)
    [[-0.61  0.34 -0.54]
     [-0.62  3.97  0.59]
     [ 4.84  3.42 -1.14]]
    >>> jnp.ceil(x)
    Array([[-0.,  1., -0.],
           [-0.,  4.,  1.],
           [ 5.,  4., -1.]], dtype=float32)
  """
  check_arraylike('ceil', x)
  if dtypes.isdtype(dtypes.dtype(x), ('integral', 'bool')):
    return lax.asarray(x)
  return lax.ceil(*promote_args_inexact('ceil', x))


@export
@partial(jit, inline=True)
def exp(x: ArrayLike, /) -> Array:
  """Calculate element-wise exponential of the input.

  JAX implementation of :obj:`numpy.exp`.

  Args:
    x: input array or scalar

  Returns:
    An array containing the exponential of each element in ``x``, promotes to
    inexact dtype.

  See also:
    - :func:`jax.numpy.log`: Calculates element-wise logarithm of the input.
    - :func:`jax.numpy.expm1`: Calculates :math:`e^x-1` of each element of the
      input.
    - :func:`jax.numpy.exp2`: Calculates base-2 exponential of each element of
      the input.

  Examples:
    ``jnp.exp`` follows the properties of exponential such as :math:`e^{(a+b)}
    = e^a * e^b`.

    >>> x1 = jnp.array([2, 4, 3, 1])
    >>> x2 = jnp.array([1, 3, 2, 3])
    >>> with jnp.printoptions(precision=2, suppress=True):
    ...   print(jnp.exp(x1+x2))
    [  20.09 1096.63  148.41   54.6 ]
    >>> with jnp.printoptions(precision=2, suppress=True):
    ...   print(jnp.exp(x1)*jnp.exp(x2))
    [  20.09 1096.63  148.41   54.6 ]

    This property holds for complex input also:

    >>> jnp.allclose(jnp.exp(3-4j), jnp.exp(3)*jnp.exp(-4j))
    Array(True, dtype=bool)
  """
  return lax.exp(*promote_args_inexact('exp', x))


@export
@partial(jit, inline=True)
def log(x: ArrayLike, /) -> Array:
  """Calculate element-wise natural logarithm of the input.

  JAX implementation of :obj:`numpy.log`.

  Args:
    x: input array or scalar.

  Returns:
    An array containing the logarithm of each element in ``x``, promotes to inexact
    dtype.

  See also:
    - :func:`jax.numpy.exp`: Calculates element-wise exponential of the input.
    - :func:`jax.numpy.log2`: Calculates base-2 logarithm of each element of input.
    - :func:`jax.numpy.log1p`: Calculates element-wise logarithm of one plus input.

  Examples:
    ``jnp.log`` and ``jnp.exp`` are inverse functions of each other. Applying
    ``jnp.log`` on the result of ``jnp.exp(x)`` yields the original input ``x``.

    >>> x = jnp.array([2, 3, 4, 5])
    >>> jnp.log(jnp.exp(x))
    Array([2., 3., 4., 5.], dtype=float32)

    Using ``jnp.log`` we can demonstrate well-known properties of logarithms, such
    as :math:`log(a*b) = log(a)+log(b)`.

    >>> x1 = jnp.array([2, 1, 3, 1])
    >>> x2 = jnp.array([1, 3, 2, 4])
    >>> jnp.allclose(jnp.log(x1*x2), jnp.log(x1)+jnp.log(x2))
    Array(True, dtype=bool)
  """
  return lax.log(*promote_args_inexact('log', x))


@export
@partial(jit, inline=True)
def expm1(x: ArrayLike, /) -> Array:
  """Calculate ``exp(x)-1`` of each element of the input.

  JAX implementation of :obj:`numpy.expm1`.

  Args:
    x: input array or scalar.

  Returns:
    An array containing ``exp(x)-1`` of each element in ``x``, promotes to inexact
    dtype.

  Note:
    ``jnp.expm1`` has much higher precision than the naive computation of
    ``exp(x)-1`` for small values of ``x``.

  See also:
    - :func:`jax.numpy.log1p`: Calculates element-wise logarithm of one plus input.
    - :func:`jax.numpy.exp`: Calculates element-wise exponential of the input.
    - :func:`jax.numpy.exp2`: Calculates base-2 exponential of each element of
      the input.

  Examples:
    >>> x = jnp.array([2, -4, 3, -1])
    >>> with jnp.printoptions(precision=2, suppress=True):
    ...   print(jnp.expm1(x))
    [ 6.39 -0.98 19.09 -0.63]
    >>> with jnp.printoptions(precision=2, suppress=True):
    ...   print(jnp.exp(x)-1)
    [ 6.39 -0.98 19.09 -0.63]

    For values very close to 0, ``jnp.expm1(x)`` is much more accurate than
    ``jnp.exp(x)-1``:

    >>> x1 = jnp.array([1e-4, 1e-6, 2e-10])
    >>> jnp.expm1(x1)
    Array([1.0000500e-04, 1.0000005e-06, 2.0000000e-10], dtype=float32)
    >>> jnp.exp(x1)-1
    Array([1.00016594e-04, 9.53674316e-07, 0.00000000e+00], dtype=float32)
  """
  return lax.expm1(*promote_args_inexact('expm1', x))


@export
@partial(jit, inline=True)
def log1p(x: ArrayLike, /) -> Array:
  """Calculates element-wise logarithm of one plus input, ``log(x+1)``.

  JAX implementation of :obj:`numpy.log1p`.

  Args:
    x: input array or scalar.

  Returns:
    An array containing the logarithm of one plus of each element in ``x``,
    promotes to inexact dtype.

  Note:
    ``jnp.log1p`` is more accurate than when using the naive computation of
    ``log(x+1)`` for small values of ``x``.

  See also:
    - :func:`jax.numpy.expm1`: Calculates :math:`e^x-1` of each element of the
      input.
    - :func:`jax.numpy.log2`: Calculates base-2 logarithm of each element of input.
    - :func:`jax.numpy.log`: Calculates element-wise logarithm of the input.

  Examples:
    >>> x = jnp.array([2, 5, 9, 4])
    >>> jnp.allclose(jnp.log1p(x), jnp.log(x+1))
    Array(True, dtype=bool)

    For values very close to 0, ``jnp.log1p(x)`` is more accurate than
    ``jnp.log(x+1)``:

    >>> x1 = jnp.array([1e-4, 1e-6, 2e-10])
    >>> jnp.expm1(jnp.log1p(x1))  # doctest: +SKIP
    Array([1.00000005e-04, 9.99999997e-07, 2.00000003e-10], dtype=float32)
    >>> jnp.expm1(jnp.log(x1+1))  # doctest: +SKIP
    Array([1.000166e-04, 9.536743e-07, 0.000000e+00], dtype=float32)
  """
  return lax.log1p(*promote_args_inexact('log1p', x))


@export
@partial(jit, inline=True)
def sin(x: ArrayLike, /) -> Array:
  """Compute a trigonometric sine of each element of input.

  JAX implementation of :obj:`numpy.sin`.

  Args:
    x: array or scalar. Angle in radians.

  Returns:
    An array containing the sine of each element in ``x``, promotes to inexact
    dtype.

  See also:
    - :func:`jax.numpy.cos`: Computes a trigonometric cosine of each element of
      input.
    - :func:`jax.numpy.tan`: Computes a trigonometric tangent of each element of
      input.
    - :func:`jax.numpy.arcsin` and :func:`jax.numpy.asin`: Computes the inverse of
      trigonometric sine of each element of input.

  Examples:
    >>> pi = jnp.pi
    >>> x = jnp.array([pi/4, pi/2, 3*pi/4, pi])
    >>> with jnp.printoptions(precision=3, suppress=True):
    ...   print(jnp.sin(x))
    [ 0.707  1.     0.707 -0.   ]
  """
  return lax.sin(*promote_args_inexact('sin', x))


@export
@partial(jit, inline=True)
def cos(x: ArrayLike, /) -> Array:
  """Compute a trigonometric cosine of each element of input.

  JAX implementation of :obj:`numpy.cos`.

  Args:
    x: scalar or array. Angle in radians.

  Returns:
    An array containing the cosine of each element in ``x``, promotes to inexact
    dtype.

  See also:
    - :func:`jax.numpy.sin`: Computes a trigonometric sine of each element of input.
    - :func:`jax.numpy.tan`: Computes a trigonometric tangent of each element of
      input.
    - :func:`jax.numpy.arccos` and :func:`jax.numpy.acos`: Computes the inverse of
      trigonometric cosine of each element of input.

  Examples:
    >>> pi = jnp.pi
    >>> x = jnp.array([pi/4, pi/2, 3*pi/4, 5*pi/6])
    >>> with jnp.printoptions(precision=3, suppress=True):
    ...   print(jnp.cos(x))
    [ 0.707 -0.    -0.707 -0.866]
  """
  return lax.cos(*promote_args_inexact('cos', x))


@export
@partial(jit, inline=True)
def tan(x: ArrayLike, /) -> Array:
  """Compute a trigonometric tangent of each element of input.

  JAX implementation of :obj:`numpy.tan`.

  Args:
    x: scalar or array. Angle in radians.

  Returns:
    An array containing the tangent of each element in ``x``, promotes to inexact
    dtype.

  See also:
    - :func:`jax.numpy.sin`: Computes a trigonometric sine of each element of input.
    - :func:`jax.numpy.cos`: Computes a trigonometric cosine of each element of
      input.
    - :func:`jax.numpy.arctan` and :func:`jax.numpy.atan`: Computes the inverse of
      trigonometric tangent of each element of input.

  Examples:
    >>> pi = jnp.pi
    >>> x = jnp.array([0, pi/6, pi/4, 3*pi/4, 5*pi/6])
    >>> with jnp.printoptions(precision=3, suppress=True):
    ...   print(jnp.tan(x))
    [ 0.     0.577  1.    -1.    -0.577]
  """
  return lax.tan(*promote_args_inexact('tan', x))


@export
@partial(jit, inline=True)
def arcsin(x: ArrayLike, /) -> Array:
  r"""Compute element-wise inverse of trigonometric sine of input.

  JAX implementation of :obj:`numpy.arcsin`.

  Args:
    x: input array or scalar.

  Returns:
    An array containing the inverse trigonometric sine of each element of ``x``
    in radians in the range ``[-pi/2, pi/2]``, promoting to inexact dtype.

  Note:
    - ``jnp.arcsin`` returns ``nan`` when ``x`` is real-valued and not in the closed
      interval ``[-1, 1]``.
    - ``jnp.arcsin`` follows the branch cut convention of :obj:`numpy.arcsin` for
      complex inputs.

  See also:
    - :func:`jax.numpy.sin`: Computes a trigonometric sine of each element of input.
    - :func:`jax.numpy.arccos` and :func:`jax.numpy.acos`: Computes the inverse of
      trigonometric cosine of each element of input.
    - :func:`jax.numpy.arctan` and :func:`jax.numpy.atan`: Computes the inverse of
      trigonometric tangent of each element of input.

  Examples:
    >>> x = jnp.array([-2, -1, -0.5, 0, 0.5, 1, 2])
    >>> with jnp.printoptions(precision=3, suppress=True):
    ...   jnp.arcsin(x)
    Array([   nan, -1.571, -0.524,  0.   ,  0.524,  1.571,    nan], dtype=float32)

    For complex-valued inputs:

    >>> with jnp.printoptions(precision=3, suppress=True):
    ...   jnp.arcsin(3+4j)
    Array(0.634+2.306j, dtype=complex64, weak_type=True)
  """
  return lax.asin(*promote_args_inexact('arcsin', x))


@export
@partial(jit, inline=True)
def arccos(x: ArrayLike, /) -> Array:
  """Compute element-wise inverse of trigonometric cosine of input.

  JAX implementation of :obj:`numpy.arccos`.

  Args:
    x: input array or scalar.

  Returns:
    An array containing the inverse trigonometric cosine of each element of ``x``
    in radians in the range ``[0, pi]``, promoting to inexact dtype.

  Note:
    - ``jnp.arccos`` returns ``nan`` when ``x`` is real-valued and not in the closed
      interval ``[-1, 1]``.
    - ``jnp.arccos`` follows the branch cut convention of :obj:`numpy.arccos` for
      complex inputs.

  See also:
    - :func:`jax.numpy.cos`: Computes a trigonometric cosine of each element of
      input.
    - :func:`jax.numpy.arcsin` and :func:`jax.numpy.asin`: Computes the inverse of
      trigonometric sine of each element of input.
    - :func:`jax.numpy.arctan` and :func:`jax.numpy.atan`: Computes the inverse of
      trigonometric tangent of each element of input.

  Examples:
    >>> x = jnp.array([-2, -1, -0.5, 0, 0.5, 1, 2])
    >>> with jnp.printoptions(precision=3, suppress=True):
    ...   jnp.arccos(x)
    Array([  nan, 3.142, 2.094, 1.571, 1.047, 0.   ,   nan], dtype=float32)

    For complex inputs:

    >>> with jnp.printoptions(precision=3, suppress=True):
    ...   jnp.arccos(4-1j)
    Array(0.252+2.097j, dtype=complex64, weak_type=True)
  """
  return lax.acos(*promote_args_inexact('arccos', x))


@export
@partial(jit, inline=True)
def arctan(x: ArrayLike, /) -> Array:
  """Compute element-wise inverse of trigonometric tangent of input.

  JAX implement of :obj:`numpy.arctan`.

  Args:
    x: input array or scalar.

  Returns:
    An array containing the inverse trigonometric tangent of each element ``x``
    in radians in the range ``[-pi/2, pi/2]``, promoting to inexact dtype.

  Note:
    ``jnp.arctan`` follows the branch cut convention of :obj:`numpy.arctan` for
    complex inputs.

  See also:
    - :func:`jax.numpy.tan`: Computes a trigonometric tangent of each element of
      input.
    - :func:`jax.numpy.arcsin` and :func:`jax.numpy.asin`: Computes the inverse of
      trigonometric sine of each element of input.
    - :func:`jax.numpy.arccos` and :func:`jax.numpy.atan`: Computes the inverse of
      trigonometric cosine of each element of input.

  Examples:
    >>> x = jnp.array([-jnp.inf, -20, -1, 0, 1, 20, jnp.inf])
    >>> with jnp.printoptions(precision=3, suppress=True):
    ...   jnp.arctan(x)
    Array([-1.571, -1.521, -0.785,  0.   ,  0.785,  1.521,  1.571], dtype=float32)

    For complex-valued inputs:

    >>> with jnp.printoptions(precision=3, suppress=True):
    ...   jnp.arctan(2+7j)
    Array(1.532+0.133j, dtype=complex64, weak_type=True)
  """
  return lax.atan(*promote_args_inexact('arctan', x))


@export
@partial(jit, inline=True)
def sinh(x: ArrayLike, /) -> Array:
  r"""Calculate element-wise hyperbolic sine of input.

  JAX implementation of :obj:`numpy.sinh`.

  The hyperbolic sine is defined by:

  .. math::

    sinh(x) = \frac{e^x - e^{-x}}{2}

  Args:
    x: input array or scalar.

  Returns:
    An array containing the hyperbolic sine of each element of ``x``, promoting
    to inexact dtype.

  Note:
    ``jnp.sinh`` is equivalent to computing ``-1j * jnp.sin(1j * x)``.

  See also:
    - :func:`jax.numpy.cosh`: Computes the element-wise hyperbolic cosine of the
      input.
    - :func:`jax.numpy.tanh`: Computes the element-wise hyperbolic tangent of the
      input.
    - :func:`jax.numpy.arcsinh`:  Computes the element-wise inverse of hyperbolic
      sine of the input.

  Examples:
    >>> x = jnp.array([[-2, 3, 5],
    ...                [0, -1, 4]])
    >>> with jnp.printoptions(precision=3, suppress=True):
    ...   jnp.sinh(x)
    Array([[-3.627, 10.018, 74.203],
           [ 0.   , -1.175, 27.29 ]], dtype=float32)
    >>> with jnp.printoptions(precision=3, suppress=True):
    ...   -1j * jnp.sin(1j * x)
    Array([[-3.627+0.j, 10.018-0.j, 74.203-0.j],
           [ 0.   -0.j, -1.175+0.j, 27.29 -0.j]],      dtype=complex64, weak_type=True)

    For complex-valued input:

    >>> with jnp.printoptions(precision=3, suppress=True):
    ...   jnp.sinh(3-2j)
    Array(-4.169-9.154j, dtype=complex64, weak_type=True)
    >>> with jnp.printoptions(precision=3, suppress=True):
    ...   -1j * jnp.sin(1j * (3-2j))
    Array(-4.169-9.154j, dtype=complex64, weak_type=True)
  """
  return lax.sinh(*promote_args_inexact('sinh', x))


@export
@partial(jit, inline=True)
def cosh(x: ArrayLike, /) -> Array:
  r"""Calculate element-wise hyperbolic cosine of input.

  JAX implementation of :obj:`numpy.cosh`.

  The hyperbolic cosine is defined by:

  .. math::

    cosh(x) = \frac{e^x + e^{-x}}{2}

  Args:
    x: input array or scalar.

  Returns:
    An array containing the hyperbolic cosine of each element of ``x``, promoting
    to inexact dtype.

  Note:
    ``jnp.cosh`` is equivalent to computing ``jnp.cos(1j * x)``.

  See also:
    - :func:`jax.numpy.sinh`: Computes the element-wise hyperbolic sine of the input.
    - :func:`jax.numpy.tanh`: Computes the element-wise hyperbolic tangent of the
      input.
    - :func:`jax.numpy.arccosh`:  Computes the element-wise inverse of hyperbolic
      cosine of the input.

  Examples:
    >>> x = jnp.array([[3, -1, 0],
    ...                [4, 7, -5]])
    >>> with jnp.printoptions(precision=3, suppress=True):
    ...   jnp.cosh(x)
    Array([[ 10.068,   1.543,   1.   ],
           [ 27.308, 548.317,  74.21 ]], dtype=float32)
    >>> with jnp.printoptions(precision=3, suppress=True):
    ...   jnp.cos(1j * x)
    Array([[ 10.068+0.j,   1.543+0.j,   1.   +0.j],
           [ 27.308+0.j, 548.317+0.j,  74.21 +0.j]],      dtype=complex64, weak_type=True)

    For complex-valued input:

    >>> with jnp.printoptions(precision=3, suppress=True):
    ...   jnp.cosh(5+1j)
    Array(40.096+62.44j, dtype=complex64, weak_type=True)
    >>> with jnp.printoptions(precision=3, suppress=True):
    ...   jnp.cos(1j * (5+1j))
    Array(40.096+62.44j, dtype=complex64, weak_type=True)
  """
  return lax.cosh(*promote_args_inexact('cosh', x))


@export
@partial(jit, inline=True)
def arcsinh(x: ArrayLike, /) -> Array:
  r"""Calculate element-wise inverse of hyperbolic sine of input.

  JAX implementation of :obj:`numpy.arcsinh`.

  The inverse of hyperbolic sine is defined by:

  .. math::

    arcsinh(x) = \ln(x + \sqrt{1 + x^2})

  Args:
    x: input array or scalar.

  Returns:
    An array of same shape as ``x`` containing the inverse of hyperbolic sine of
    each element of ``x``, promoting to inexact dtype.

  Note:
    - ``jnp.arcsinh`` returns ``nan`` for values outside the range ``(-inf, inf)``.
    - ``jnp.arcsinh`` follows the branch cut convention of :obj:`numpy.arcsinh`
      for complex inputs.

  See also:
    - :func:`jax.numpy.sinh`: Computes the element-wise hyperbolic sine of the input.
    - :func:`jax.numpy.arccosh`: Computes the element-wise inverse of hyperbolic
      cosine of the input.
    - :func:`jax.numpy.arctanh`: Computes the element-wise inverse of hyperbolic
      tangent of the input.

  Examples:
    >>> x = jnp.array([[-2, 3, 1],
    ...                [4, 9, -5]])
    >>> with jnp.printoptions(precision=3, suppress=True):
    ...   jnp.arcsinh(x)
    Array([[-1.444,  1.818,  0.881],
           [ 2.095,  2.893, -2.312]], dtype=float32)

    For complex-valued inputs:

    >>> x1 = jnp.array([4-3j, 2j])
    >>> with jnp.printoptions(precision=3, suppress=True):
    ...   jnp.arcsinh(x1)
    Array([2.306-0.634j, 1.317+1.571j], dtype=complex64)
  """
  return lax.asinh(*promote_args_inexact('arcsinh', x))


@export
@jit
def arccosh(x: ArrayLike, /) -> Array:
  r"""Calculate element-wise inverse of hyperbolic cosine of input.

  JAX implementation of :obj:`numpy.arccosh`.

  The inverse of hyperbolic cosine is defined by:

  .. math::

    arccosh(x) = \ln(x + \sqrt{x^2 - 1})

  Args:
    x: input array or scalar.

  Returns:
    An array of same shape as ``x`` containing the inverse of hyperbolic cosine
    of each element of ``x``, promoting to inexact dtype.

  Note:
    - ``jnp.arccosh`` returns ``nan`` for real-values in the range ``[-inf, 1)``.
    - ``jnp.arccosh`` follows the branch cut convention of :obj:`numpy.arccosh`
      for complex inputs.

  See also:
    - :func:`jax.numpy.cosh`: Computes the element-wise hyperbolic cosine of the
      input.
    - :func:`jax.numpy.arcsinh`: Computes the element-wise inverse of hyperbolic
      sine of the input.
    - :func:`jax.numpy.arctanh`: Computes the element-wise inverse of hyperbolic
      tangent of the input.

  Examples:
    >>> x = jnp.array([[1, 3, -4],
    ...                [-5, 2, 7]])
    >>> with jnp.printoptions(precision=3, suppress=True):
    ...   jnp.arccosh(x)
    Array([[0.   , 1.763,   nan],
           [  nan, 1.317, 2.634]], dtype=float32)

    For complex-valued input:

    >>> x1 = jnp.array([-jnp.inf+0j, 1+2j, -5+0j])
    >>> with jnp.printoptions(precision=3, suppress=True):
    ...   jnp.arccosh(x1)
    Array([  inf+3.142j, 1.529+1.144j, 2.292+3.142j], dtype=complex64)
  """
  # Note: arccosh is multi-valued for complex input, and lax.acosh
  # uses a different convention than np.arccosh.
  result = lax.acosh(*promote_args_inexact("arccosh", x))
  if dtypes.issubdtype(result.dtype, np.complexfloating):
    result = _where(real(result) < 0, lax.neg(result), result)
  return result


@export
@partial(jit, inline=True)
def tanh(x: ArrayLike, /) -> Array:
  r"""Calculate element-wise hyperbolic tangent of input.

  JAX implementation of :obj:`numpy.tanh`.

  The hyperbolic tangent is defined by:

  .. math::

    tanh(x) = \frac{sinh(x)}{cosh(x)} = \frac{e^x - e^{-x}}{e^x + e^{-x}}

  Args:
    x: input array or scalar.

  Returns:
    An array containing the hyperbolic tangent of each element of ``x``, promoting
    to inexact dtype.

  Note:
    ``jnp.tanh`` is equivalent to computing ``-1j * jnp.tan(1j * x)``.

  See also:
    - :func:`jax.numpy.sinh`: Computes the element-wise hyperbolic sine of the input.
    - :func:`jax.numpy.cosh`: Computes the element-wise hyperbolic cosine of the
      input.
    - :func:`jax.numpy.arctanh`:  Computes the element-wise inverse of hyperbolic
      tangent of the input.

  Examples:
    >>> x = jnp.array([[-1, 0, 1],
    ...                [3, -2, 5]])
    >>> with jnp.printoptions(precision=3, suppress=True):
    ...   jnp.tanh(x)
    Array([[-0.762,  0.   ,  0.762],
           [ 0.995, -0.964,  1.   ]], dtype=float32)
    >>> with jnp.printoptions(precision=3, suppress=True):
    ...   -1j * jnp.tan(1j * x)
    Array([[-0.762+0.j,  0.   -0.j,  0.762-0.j],
           [ 0.995-0.j, -0.964+0.j,  1.   -0.j]],      dtype=complex64, weak_type=True)

    For complex-valued input:

    >>> with jnp.printoptions(precision=3, suppress=True):
    ...   jnp.tanh(2-5j)
    Array(1.031+0.021j, dtype=complex64, weak_type=True)
    >>> with jnp.printoptions(precision=3, suppress=True):
    ...   -1j * jnp.tan(1j * (2-5j))
    Array(1.031+0.021j, dtype=complex64, weak_type=True)
  """
  return lax.tanh(*promote_args_inexact('tanh', x))


@export
@partial(jit, inline=True)
def arctanh(x: ArrayLike, /) -> Array:
  r"""Calculate element-wise inverse of hyperbolic tangent of input.

  JAX implementation of :obj:`numpy.arctanh`.

  The inverse of hyperbolic tangent is defined by:

  .. math::

    arctanh(x) = \frac{1}{2} [\ln(1 + x) - \ln(1 - x)]

  Args:
    x: input array or scalar.

  Returns:
    An array of same shape as ``x`` containing the inverse of hyperbolic tangent
    of each element of ``x``, promoting to inexact dtype.

  Note:
    - ``jnp.arctanh`` returns ``nan`` for real-values outside the range ``[-1, 1]``.
    - ``jnp.arctanh`` follows the branch cut convention of :obj:`numpy.arctanh`
      for complex inputs.

  See also:
    - :func:`jax.numpy.tanh`: Computes the element-wise hyperbolic tangent of the
      input.
    - :func:`jax.numpy.arcsinh`: Computes the element-wise inverse of hyperbolic
      sine of the input.
    - :func:`jax.numpy.arccosh`: Computes the element-wise inverse of hyperbolic
      cosine of the input.

  Examples:
    >>> x = jnp.array([-2, -1, -0.5, 0, 0.5, 1, 2])
    >>> with jnp.printoptions(precision=3, suppress=True):
    ...   jnp.arctanh(x)
    Array([   nan,   -inf, -0.549,  0.   ,  0.549,    inf,    nan], dtype=float32)

    For complex-valued input:

    >>> x1 = jnp.array([-2+0j, 3+0j, 4-1j])
    >>> with jnp.printoptions(precision=3, suppress=True):
    ...   jnp.arctanh(x1)
    Array([-0.549+1.571j,  0.347+1.571j,  0.239-1.509j], dtype=complex64)
  """
  return lax.atanh(*promote_args_inexact('arctanh', x))


@export
@partial(jit, inline=True)
def sqrt(x: ArrayLike, /) -> Array:
  """Calculates element-wise non-negative square root of the input array.

  JAX implementation of :obj:`numpy.sqrt`.

  Args:
    x: input array or scalar.

  Returns:
    An array containing the non-negative square root of the elements of ``x``.

  Note:
    - For real-valued negative inputs, ``jnp.sqrt`` produces a ``nan`` output.
    - For complex-valued negative inputs, ``jnp.sqrt`` produces a ``complex`` output.

  See also:
    - :func:`jax.numpy.square`: Calculates the element-wise square of the input.
    - :func:`jax.numpy.power`: Calculates the element-wise base ``x1`` exponential
      of ``x2``.

  Examples:
    >>> x = jnp.array([-8-6j, 1j, 4])
    >>> with jnp.printoptions(precision=3, suppress=True):
    ...   jnp.sqrt(x)
    Array([1.   -3.j   , 0.707+0.707j, 2.   +0.j   ], dtype=complex64)
    >>> jnp.sqrt(-1)
    Array(nan, dtype=float32, weak_type=True)
  """
  return lax.sqrt(*promote_args_inexact('sqrt', x))


@export
@partial(jit, inline=True)
def cbrt(x: ArrayLike, /) -> Array:
  """Calculates element-wise cube root of the input array.

  JAX implementation of :obj:`numpy.cbrt`.

  Args:
    x: input array or scalar. ``complex`` dtypes are not supported.

  Returns:
    An array containing the cube root of the elements of ``x``.

  See also:
    - :func:`jax.numpy.sqrt`: Calculates the element-wise non-negative square root
      of the input.
    - :func:`jax.numpy.square`: Calculates the element-wise square of the input.

  Examples:
    >>> x = jnp.array([[216, 125, 64],
    ...                [-27, -8, -1]])
    >>> with jnp.printoptions(precision=3, suppress=True):
    ...   jnp.cbrt(x)
    Array([[ 6.,  5.,  4.],
           [-3., -2., -1.]], dtype=float32)
  """
  return lax.cbrt(*promote_args_inexact('cbrt', x))


def _add_at(a: Array, indices: Any, b: ArrayLike) -> Array:
  """Implementation of jnp.add.at."""
  if a.dtype == bool:
    a = a.astype('int32')
    b = lax.convert_element_type(b, bool).astype('int32')
    return a.at[indices].add(b).astype(bool)
  return a.at[indices].add(b)


@binary_ufunc(identity=0, reduce=reductions.sum, accumulate=reductions.cumsum, at=_add_at)
def add(x: ArrayLike, y: ArrayLike, /) -> Array:
  """Add two arrays element-wise.

  JAX implementation of :obj:`numpy.add`. This is a universal function,
  and supports the additional APIs described at :class:`jax.numpy.ufunc`.
  This function provides the implementation of the ``+`` operator for
  JAX arrays.

  Args:
    x, y: arrays to add. Must be broadcastable to a common shape.

  Returns:
    Array containing the result of the element-wise addition.

  Examples:
    Calling ``add`` explicitly:

    >>> x = jnp.arange(4)
    >>> jnp.add(x, 10)
    Array([10, 11, 12, 13], dtype=int32)

    Calling ``add`` via the ``+`` operator:

    >>> x + 10
    Array([10, 11, 12, 13], dtype=int32)
  """
  x, y = promote_args("add", x, y)
  return lax.add(x, y) if x.dtype != bool else lax.bitwise_or(x, y)


def _multiply_at(a: Array, indices: Any, b: ArrayLike) -> Array:
  """Implementation of jnp.multiply.at."""
  if a.dtype == bool:
    a = a.astype('int32')
    b = lax.convert_element_type(b, bool).astype('int32')
    return a.at[indices].mul(b).astype(bool)
  else:
    return a.at[indices].mul(b)


@binary_ufunc(identity=1, reduce=reductions.prod, accumulate=reductions.cumprod, at=_multiply_at)
def multiply(x: ArrayLike, y: ArrayLike, /) -> Array:
  """Multiply two arrays element-wise.

  JAX implementation of :obj:`numpy.multiply`. This is a universal function,
  and supports the additional APIs described at :class:`jax.numpy.ufunc`.
  This function provides the implementation of the ``*`` operator for
  JAX arrays.

  Args:
    x, y: arrays to multiply. Must be broadcastable to a common shape.

  Returns:
    Array containing the result of the element-wise multiplication.

  Examples:
    Calling ``multiply`` explicitly:

    >>> x = jnp.arange(4)
    >>> jnp.multiply(x, 10)
    Array([ 0, 10, 20, 30], dtype=int32)

    Calling ``multiply`` via the ``*`` operator:

    >>> x * 10
    Array([ 0, 10, 20, 30], dtype=int32)
  """
  x, y = promote_args("multiply", x, y)
  return lax.mul(x, y) if x.dtype != bool else lax.bitwise_and(x, y)


@binary_ufunc(identity=-1, reduce=reductions._reduce_bitwise_and)
def bitwise_and(x: ArrayLike, y: ArrayLike, /) -> Array:
  """Compute the bitwise AND operation elementwise.

  JAX implementation of :obj:`numpy.bitwise_and`. This is a universal function,
  and supports the additional APIs described at :class:`jax.numpy.ufunc`.
  This function provides the implementation of the ``&`` operator for
  JAX arrays.

  Args:
    x, y: integer or boolean arrays. Must be broadcastable to a common shape.

  Returns:
    Array containing the result of the element-wise bitwise AND.

  Examples:
    Calling ``bitwise_and`` explicitly:

    >>> x = jnp.arange(4)
    >>> jnp.bitwise_and(x, 1)
    Array([0, 1, 0, 1], dtype=int32)

    Calling ``bitwise_and`` via the ``&`` operator:

    >>> x & 1
    Array([0, 1, 0, 1], dtype=int32)
  """
  return lax.bitwise_and(*promote_args("bitwise_and", x, y))


@binary_ufunc(identity=0, reduce=reductions._reduce_bitwise_or)
def bitwise_or(x: ArrayLike, y: ArrayLike, /) -> Array:
  """Compute the bitwise OR operation elementwise.

  JAX implementation of :obj:`numpy.bitwise_or`. This is a universal function,
  and supports the additional APIs described at :class:`jax.numpy.ufunc`.
  This function provides the implementation of the ``|`` operator for
  JAX arrays.

  Args:
    x, y: integer or boolean arrays. Must be broadcastable to a common shape.

  Returns:
    Array containing the result of the element-wise bitwise OR.

  Examples:
    Calling ``bitwise_or`` explicitly:

    >>> x = jnp.arange(4)
    >>> jnp.bitwise_or(x, 1)
    Array([1, 1, 3, 3], dtype=int32)

    Calling ``bitwise_or`` via the ``|`` operator:

    >>> x | 1
    Array([1, 1, 3, 3], dtype=int32)
  """
  return lax.bitwise_or(*promote_args("bitwise_or", x, y))


@binary_ufunc(identity=0, reduce=reductions._reduce_bitwise_xor)
def bitwise_xor(x: ArrayLike, y: ArrayLike, /) -> Array:
  """Compute the bitwise XOR operation elementwise.

  JAX implementation of :obj:`numpy.bitwise_xor`. This is a universal function,
  and supports the additional APIs described at :class:`jax.numpy.ufunc`.
  This function provides the implementation of the ``^`` operator for
  JAX arrays.

  Args:
    x, y: integer or boolean arrays. Must be broadcastable to a common shape.

  Returns:
    Array containing the result of the element-wise bitwise XOR.

  Examples:
    Calling ``bitwise_xor`` explicitly:

    >>> x = jnp.arange(4)
    >>> jnp.bitwise_xor(x, 1)
    Array([1, 0, 3, 2], dtype=int32)

    Calling ``bitwise_xor`` via the ``^`` operator:

    >>> x ^ 1
    Array([1, 0, 3, 2], dtype=int32)
  """
  return lax.bitwise_xor(*promote_args("bitwise_xor", x, y))


@export
@partial(jit, inline=True)
def left_shift(x: ArrayLike, y: ArrayLike, /) -> Array:
  r"""Shift bits of ``x`` to left by the amount specified in ``y``, element-wise.

  JAX implementation of :obj:`numpy.left_shift`.

  Args:
    x: Input array, must be integer-typed.
    y: The amount of bits to shift each element in ``x`` to the left, only accepts
      integer subtypes. ``x`` and ``y`` must either have same shape or be broadcast
      compatible.

  Returns:
    An array containing the left shifted elements of ``x`` by the amount specified
    in ``y``, with the same shape as the broadcasted shape of ``x`` and ``y``.

  Note:
    Left shifting ``x`` by ``y`` is equivalent to ``x * (2**y)`` within the
    bounds of the dtypes involved.

  See also:
    - :func:`jax.numpy.right_shift`: and :func:`jax.numpy.bitwise_right_shift`:
      Shifts the bits of ``x1`` to right by the amount specified in ``x2``,
      element-wise.
    - :func:`jax.numpy.bitwise_left_shift`: Alias of :func:`jax.left_shift`.

  Examples:
    >>> def print_binary(x):
    ...   return [bin(int(val)) for val in x]

    >>> x1 = jnp.arange(5)
    >>> x1
    Array([0, 1, 2, 3, 4], dtype=int32)
    >>> print_binary(x1)
    ['0b0', '0b1', '0b10', '0b11', '0b100']
    >>> x2 = 1
    >>> result = jnp.left_shift(x1, x2)
    >>> result
    Array([0, 2, 4, 6, 8], dtype=int32)
    >>> print_binary(result)
    ['0b0', '0b10', '0b100', '0b110', '0b1000']

    >>> x3 = 4
    >>> print_binary([x3])
    ['0b100']
    >>> x4 = jnp.array([1, 2, 3, 4])
    >>> result1 = jnp.left_shift(x3, x4)
    >>> result1
    Array([ 8, 16, 32, 64], dtype=int32)
    >>> print_binary(result1)
    ['0b1000', '0b10000', '0b100000', '0b1000000']
  """
  return lax.shift_left(*promote_args_numeric("left_shift", x, y))


@export
@partial(jit, inline=True)
def bitwise_left_shift(x: ArrayLike, y: ArrayLike, /) -> Array:
  """Alias of :func:`jax.numpy.left_shift`."""
  return lax.shift_left(*promote_args_numeric("bitwise_left_shift", x, y))


@export
@partial(jit, inline=True)
def equal(x: ArrayLike, y: ArrayLike, /) -> Array:
  """Returns element-wise truth value of ``x == y``.

  JAX implementation of :obj:`numpy.equal`. This function provides the implementation
  of the ``==`` operator for JAX arrays.

  Args:
    x: input array or scalar.
    y: input array or scalar. ``x`` and ``y`` should either have same shape or be
      broadcast compatible.

  Returns:
    A boolean array containing ``True`` where the elements of ``x == y`` and
    ``False`` otherwise.

  See also:
    - :func:`jax.numpy.not_equal`: Returns element-wise truth value of ``x != y``.
    - :func:`jax.numpy.greater_equal`: Returns element-wise truth value of
      ``x >= y``.
    - :func:`jax.numpy.less_equal`: Returns element-wise truth value of ``x <= y``.
    - :func:`jax.numpy.greater`: Returns element-wise truth value of ``x > y``.
    - :func:`jax.numpy.less`: Returns element-wise truth value of ``x < y``.

  Examples:
    >>> jnp.equal(0., -0.)
    Array(True, dtype=bool, weak_type=True)
    >>> jnp.equal(1, 1.)
    Array(True, dtype=bool, weak_type=True)
    >>> jnp.equal(5, jnp.array(5))
    Array(True, dtype=bool, weak_type=True)
    >>> jnp.equal(2, -2)
    Array(False, dtype=bool, weak_type=True)
    >>> x = jnp.array([[1, 2, 3],
    ...                [4, 5, 6],
    ...                [7, 8, 9]])
    >>> y = jnp.array([1, 5, 9])
    >>> jnp.equal(x, y)
    Array([[ True, False, False],
           [False,  True, False],
           [False, False,  True]], dtype=bool)
    >>> x == y
    Array([[ True, False, False],
           [False,  True, False],
           [False, False,  True]], dtype=bool)
  """
  return lax.eq(*promote_args("equal", x, y))


@export
@partial(jit, inline=True)
def not_equal(x: ArrayLike, y: ArrayLike, /) -> Array:
  """Returns element-wise truth value of ``x != y``.

  JAX implementation of :obj:`numpy.not_equal`. This function provides the
  implementation of the ``!=`` operator for JAX arrays.

  Args:
    x: input array or scalar.
    y: input array or scalar. ``x`` and ``y`` should either have same shape or be
      broadcast compatible.

  Returns:
    A boolean array containing ``True`` where the elements of ``x != y`` and
    ``False`` otherwise.

  See also:
    - :func:`jax.numpy.equal`: Returns element-wise truth value of ``x == y``.
    - :func:`jax.numpy.greater_equal`: Returns element-wise truth value of
      ``x >= y``.
    - :func:`jax.numpy.less_equal`: Returns element-wise truth value of ``x <= y``.
    - :func:`jax.numpy.greater`: Returns element-wise truth value of ``x > y``.
    - :func:`jax.numpy.less`: Returns element-wise truth value of ``x < y``.

  Examples:
    >>> jnp.not_equal(0., -0.)
    Array(False, dtype=bool, weak_type=True)
    >>> jnp.not_equal(-2, 2)
    Array(True, dtype=bool, weak_type=True)
    >>> jnp.not_equal(1, 1.)
    Array(False, dtype=bool, weak_type=True)
    >>> jnp.not_equal(5, jnp.array(5))
    Array(False, dtype=bool, weak_type=True)
    >>> x = jnp.array([[1, 2, 3],
    ...                [4, 5, 6],
    ...                [7, 8, 9]])
    >>> y = jnp.array([1, 5, 9])
    >>> jnp.not_equal(x, y)
    Array([[False,  True,  True],
           [ True, False,  True],
           [ True,  True, False]], dtype=bool)
    >>> x != y
    Array([[False,  True,  True],
           [ True, False,  True],
           [ True,  True, False]], dtype=bool)
  """
  return lax.ne(*promote_args("not_equal", x, y))


def _subtract_at(a: Array, indices: Any, b: ArrayLike) -> Array:
  """Implementation of jnp.subtract.at."""
  return a.at[indices].subtract(b)


@binary_ufunc(identity=None, at=_subtract_at)
def subtract(x: ArrayLike, y: ArrayLike, /) -> Array:
  """Subtract two arrays element-wise.

  JAX implementation of :obj:`numpy.subtract`. This is a universal function,
  and supports the additional APIs described at :class:`jax.numpy.ufunc`.
  This function provides the implementation of the ``-`` operator for
  JAX arrays.

  Args:
    x, y: arrays to subtract. Must be broadcastable to a common shape.

  Returns:
    Array containing the result of the element-wise subtraction.

  Examples:
    Calling ``subtract`` explicitly:

    >>> x = jnp.arange(4)
    >>> jnp.subtract(x, 10)
    Array([-10,  -9,  -8,  -7], dtype=int32)

    Calling ``subtract`` via the ``-`` operator:

    >>> x - 10
    Array([-10,  -9,  -8,  -7], dtype=int32)
  """
  return lax.sub(*promote_args("subtract", x, y))


@export
@partial(jit, inline=True)
def arctan2(x1: ArrayLike, x2: ArrayLike, /) -> Array:
  r"""Compute the arctangent of x1/x2, choosing the correct quadrant.

  JAX implementation of :func:`numpy.arctan2`

  Args:
    x1: numerator array.
    x2: denomniator array; should be broadcast-compatible with x1.

  Returns:
    The elementwise arctangent of x1 / x2, tracking the correct quadrant.

  See also:
    - :func:`jax.numpy.tan`: compute the tangent of an angle
    - :func:`jax.numpy.atan2`: the array API version of this function.

  Examples:
    Consider a sequence of angles in radians between 0 and :math:`2\pi`:

    >>> theta = jnp.linspace(-jnp.pi, jnp.pi, 9)
    >>> with jnp.printoptions(precision=2, suppress=True):
    ...   print(theta)
    [-3.14 -2.36 -1.57 -0.79  0.    0.79  1.57  2.36  3.14]

    These angles can equivalently be represented by ``(x, y)`` coordinates
    on a unit circle:

    >>> x, y = jnp.cos(theta), jnp.sin(theta)

    To reconstruct the input angle, we might be tempted to use the identity
    :math:`\tan(\theta) = y / x`, and compute :math:`\theta = \tan^{-1}(y/x)`.
    Unfortunately, this does not recover the input angle:

    >>> with jnp.printoptions(precision=2, suppress=True):
    ...   print(jnp.arctan(y / x))
    [-0.    0.79  1.57 -0.79  0.    0.79  1.57 -0.79  0.  ]

    The problem is that :math:`y/x` contains some ambiguity: although
    :math:`(y, x) = (-1, -1)` and :math:`(y, x) = (1, 1)` represent different points in
    Cartesian space, in both cases :math:`y / x = 1`, and so the simple arctan
    approach loses information about which quadrant the angle lies in. :func:`arctan2`
    is built to address this:

    >>> with jnp.printoptions(precision=2, suppress=True):
    ...  print(jnp.arctan2(y, x))
    [ 3.14 -2.36 -1.57 -0.79  0.    0.79  1.57  2.36 -3.14]

    The results match the input ``theta``, except at the endpoints where :math:`+\pi`
    and :math:`-\pi` represent indistinguishable points on the unit circle. By convention,
    :func:`arctan2` alwasy returns values between :math:`-\pi` and :math:`+\pi` inclusive.
  """
  return lax.atan2(*promote_args_inexact("arctan2", x1, x2))


@export
@partial(jit, inline=True)
def minimum(x: ArrayLike, y: ArrayLike, /) -> Array:
  """Return element-wise minimum of the input arrays.

  JAX implementation of :obj:`numpy.minimum`.

  Args:
    x: input array or scalar.
    y: input array or scalar. Both ``x`` and ``y`` should either have same shape
      or be broadcast compatible.

  Returns:
    An array containing the element-wise minimum of ``x`` and ``y``.

  Note:
    For each pair of elements, ``jnp.minimum`` returns:
      - smaller of the two if both elements are finite numbers.
      - ``nan`` if one element is ``nan``.

  See also:
    - :func:`jax.numpy.maximum`: Returns element-wise maximum of the input arrays.
    - :func:`jax.numpy.fmin`: Returns element-wise minimum of the input arrays,
      ignoring NaNs.
    - :func:`jax.numpy.amin`: Returns the minimum of array elements along a given
      axis.
    - :func:`jax.numpy.nanmin`: Returns the minimum of the array elements along
      a given axis, ignoring NaNs.

  Examples:
    Inputs with ``x.shape == y.shape``:

    >>> x = jnp.array([2, 3, 5, 1])
    >>> y = jnp.array([-3, 6, -4, 7])
    >>> jnp.minimum(x, y)
    Array([-3,  3, -4,  1], dtype=int32)

    Inputs having broadcast compatibility:

    >>> x1 = jnp.array([[1, 5, 2],
    ...                 [-3, 4, 7]])
    >>> y1 = jnp.array([-2, 3, 6])
    >>> jnp.minimum(x1, y1)
    Array([[-2,  3,  2],
           [-3,  3,  6]], dtype=int32)

    Inputs with ``nan``:

    >>> nan = jnp.nan
    >>> x2 = jnp.array([[2.5, nan, -2],
    ...                 [nan, 5, 6],
    ...                 [-4, 3, 7]])
    >>> y2 = jnp.array([1, nan, 5])
    >>> jnp.minimum(x2, y2)
    Array([[ 1., nan, -2.],
           [nan, nan,  5.],
           [-4., nan,  5.]], dtype=float32)
  """
  return lax.min(*promote_args("minimum", x, y))


@export
@partial(jit, inline=True)
def maximum(x: ArrayLike, y: ArrayLike, /) -> Array:
  """Return element-wise maximum of the input arrays.

  JAX implementation of :obj:`numpy.maximum`.

  Args:
    x: input array or scalar.
    y: input array or scalar. Both ``x`` and ``y`` should either have same shape
      or be broadcast compatible.

  Returns:
    An array containing the element-wise maximum of ``x`` and ``y``.

  Note:
    For each pair of elements, ``jnp.maximum`` returns:
      - larger of the two if both elements are finite numbers.
      - ``nan`` if one element is ``nan``.

  See also:
    - :func:`jax.numpy.minimum`: Returns element-wise minimum of the input
      arrays.
    - :func:`jax.numpy.fmax`: Returns element-wise maximum of the input arrays,
      ignoring NaNs.
    - :func:`jax.numpy.amax`: Retruns the maximum of array elements along a given
      axis.
    - :func:`jax.numpy.nanmax`: Returns the maximum of the array elements along
      a given axis, ignoring NaNs.

  Examples:
    Inputs with ``x.shape == y.shape``:

    >>> x = jnp.array([1, -5, 3, 2])
    >>> y = jnp.array([-2, 4, 7, -6])
    >>> jnp.maximum(x, y)
    Array([1, 4, 7, 2], dtype=int32)

    Inputs with broadcast compatibility:

    >>> x1 = jnp.array([[-2, 5, 7, 4],
    ...                 [1, -6, 3, 8]])
    >>> y1 = jnp.array([-5, 3, 6, 9])
    >>> jnp.maximum(x1, y1)
    Array([[-2,  5,  7,  9],
           [ 1,  3,  6,  9]], dtype=int32)

    Inputs having ``nan``:

    >>> nan = jnp.nan
    >>> x2 = jnp.array([nan, -3, 9])
    >>> y2 = jnp.array([[4, -2, nan],
    ...                 [-3, -5, 10]])
    >>> jnp.maximum(x2, y2)
    Array([[nan, -2., nan],
          [nan, -3., 10.]], dtype=float32)
  """
  return lax.max(*promote_args("maximum", x, y))


@export
@partial(jit, inline=True)
def float_power(x: ArrayLike, y: ArrayLike, /) -> Array:
  """Calculate element-wise base ``x`` exponential of ``y``.

  JAX implementation of :obj:`numpy.float_power`.

  Args:
    x: scalar or array. Specifies the bases.
    y: scalar or array. Specifies the exponents. ``x`` and ``y`` should either
      have same shape or be broadcast compatible.

  Returns:
    An array containing the base ``x`` exponentials of ``y``, promoting to the
    inexact dtype.

  See also:
    - :func:`jax.numpy.exp`: Calculates element-wise exponential of the input.
    - :func:`jax.numpy.exp2`: Calculates base-2 exponential of each element of
      the input.

  Examples:
    Inputs with same shape:

    >>> x = jnp.array([3, 1, -5])
    >>> y = jnp.array([2, 4, -1])
    >>> jnp.float_power(x, y)
    Array([ 9. ,  1. , -0.2], dtype=float32)

    Inputs with broacast compatibility:

    >>> x1 = jnp.array([[2, -4, 1],
    ...                 [-1, 2, 3]])
    >>> y1 = jnp.array([-2, 1, 4])
    >>> jnp.float_power(x1, y1)
    Array([[ 0.25, -4.  ,  1.  ],
           [ 1.  ,  2.  , 81.  ]], dtype=float32)

    ``jnp.float_power`` produces ``nan`` for negative values raised to a non-integer
    values.

    >>> jnp.float_power(-3, 1.7)
    Array(nan, dtype=float32, weak_type=True)
  """
  return lax.pow(*promote_args_inexact("float_power", x, y))


@export
@partial(jit, inline=True)
def nextafter(x: ArrayLike, y: ArrayLike, /) -> Array:
  """Return element-wise next floating point value after ``x`` towards ``y``.

  JAX implementation of :obj:`numpy.nextafter`.

  Args:
    x: scalar or array. Specifies the value after which the next number is found.
    y: scalar or array. Specifies the direction towards which the next number is
      found. ``x`` and ``y`` should either have same shape or be broadcast
      compatible.

  Returns:
    An array containing the next representable number of ``x`` in the direction
    of ``y``.

  Examples:
    >>> jnp.nextafter(2, 1)  # doctest: +SKIP
    Array(1.9999999, dtype=float32, weak_type=True)
    >>> x = jnp.array([3, -2, 1])
    >>> y = jnp.array([2, -1, 2])
    >>> jnp.nextafter(x, y)  # doctest: +SKIP
    Array([ 2.9999998, -1.9999999,  1.0000001], dtype=float32)
  """
  return lax.nextafter(*promote_args_inexact("nextafter", x, y))


@export
@partial(jit, inline=True)
def spacing(x: ArrayLike, /) -> Array:
  """Return the spacing between ``x`` and the next adjacent number.

  JAX implementation of :func:`numpy.spacing`.

  Args:
    x: real-valued array. Integer or boolean types will be cast to float.

  Returns:
    Array of same shape as ``x`` containing spacing between each entry of
    ``x`` and its closest adjacent value.

  See also:
    - :func:`jax.numpy.nextafter`: find the next representable value.

  Examples:
    >>> x = jnp.array([0.0, 0.25, 0.5, 0.75, 1.0], dtype='float32')
    >>> jnp.spacing(x)
    Array([1.4012985e-45, 2.9802322e-08, 5.9604645e-08, 5.9604645e-08,
          1.1920929e-07], dtype=float32)

    For ``x = 1``, the spacing is equal to the ``eps`` value given by
    :class:`jax.numpy.finfo`:

    >>> x = jnp.float32(1)
    >>> jnp.spacing(x) == jnp.finfo(x.dtype).eps
    Array(True, dtype=bool)
  """
  arr, = promote_args_inexact("spacing", x)
  if dtypes.isdtype(arr.dtype, "complex floating"):
    raise ValueError("jnp.spacing is not defined for complex inputs.")
  inf = _lax_const(arr, np.inf)
  smallest_subnormal = dtypes.finfo(arr.dtype).smallest_subnormal

  # Numpy's behavior seems to depend on dtype
  if arr.dtype == 'float16':
    return lax.nextafter(arr, inf) - arr
  else:
    result = lax.nextafter(arr, copysign(inf, arr)) - arr
    return _where(result == 0, copysign(smallest_subnormal, arr), result)


# Logical ops
@binary_ufunc(identity=True, reduce=reductions._reduce_logical_and)
def logical_and(x: ArrayLike, y: ArrayLike, /) -> Array:
  """Compute the logical AND operation elementwise.

  JAX implementation of :obj:`numpy.logical_and`. This is a universal function,
  and supports the additional APIs described at :class:`jax.numpy.ufunc`.

  Args:
    x, y: input arrays. Must be broadcastable to a common shape.

  Returns:
    Array containing the result of the element-wise logical AND.

  Examples:
    >>> x = jnp.arange(4)
    >>> jnp.logical_and(x, 1)
    Array([False,  True,  True,  True], dtype=bool)
  """
  return lax.bitwise_and(*map(_to_bool, promote_args("logical_and", x, y)))


@binary_ufunc(identity=False, reduce=reductions._reduce_logical_or)
def logical_or(x: ArrayLike, y: ArrayLike, /) -> Array:
  """Compute the logical OR operation elementwise.

  JAX implementation of :obj:`numpy.logical_or`. This is a universal function,
  and supports the additional APIs described at :class:`jax.numpy.ufunc`.

  Args:
    x, y: input arrays. Must be broadcastable to a common shape.

  Returns:
    Array containing the result of the element-wise logical OR.

  Examples:
    >>> x = jnp.arange(4)
    >>> jnp.logical_or(x, 1)
    Array([ True,  True,  True,  True], dtype=bool)
  """
  return lax.bitwise_or(*map(_to_bool, promote_args("logical_or", x, y)))


@binary_ufunc(identity=False, reduce=reductions._reduce_logical_xor)
def logical_xor(x: ArrayLike, y: ArrayLike, /) -> Array:
  """Compute the logical XOR operation elementwise.

  JAX implementation of :obj:`numpy.logical_xor`. This is a universal function,
  and supports the additional APIs described at :class:`jax.numpy.ufunc`.

  Args:
    x, y: input arrays. Must be broadcastable to a common shape.

  Returns:
    Array containing the result of the element-wise logical XOR.

  Examples:
    >>> x = jnp.arange(4)
    >>> jnp.logical_xor(x, 1)
    Array([ True, False, False, False], dtype=bool)
  """
  return lax.bitwise_xor(*map(_to_bool, promote_args("logical_xor", x, y)))


@export
@partial(jit, inline=True)
def logical_not(x: ArrayLike, /) -> Array:
  """Compute NOT bool(x) element-wise.

  JAX implementation of :func:`numpy.logical_not`.

  Args:
    x: input array of any dtype.

  Returns:
    A boolean array that computes NOT bool(x) element-wise

  See also:
    - :func:`jax.numpy.invert` or :func:`jax.numpy.bitwise_invert`: bitwise NOT operation

  Examples:
    Compute NOT x element-wise on a boolean array:

    >>> x = jnp.array([True, False, True])
    >>> jnp.logical_not(x)
    Array([False,  True, False], dtype=bool)

    For boolean input, this is equivalent to :func:`~jax.numpy.invert`, which implements
    the unary ``~`` operator:

    >>> ~x
    Array([False,  True, False], dtype=bool)

    For non-boolean input, the input of :func:`logical_not` is implicitly cast to boolean:

    >>> x = jnp.array([-1, 0, 1])
    >>> jnp.logical_not(x)
    Array([False,  True, False], dtype=bool)
  """
  return lax.bitwise_not(*map(_to_bool, promote_args("logical_not", x)))

# Comparison ops
def _complex_comparison(lax_op: Callable[[ArrayLike, ArrayLike], Array],
                        x: Array, y: Array):
  if dtypes.issubdtype(x.dtype, np.complexfloating):
    return lax.select(lax.eq(x.real, y.real),
                      lax_op(x.imag, y.imag),
                      lax_op(x.real, y.real))
  return lax_op(x, y)


@export
@partial(jit, inline=True)
def greater_equal(x: ArrayLike, y: ArrayLike, /) -> Array:
  """Return element-wise truth value of ``x >= y``.

  JAX implementation of :obj:`numpy.greater_equal`.

  Args:
    x: input array or scalar.
    y: input array or scalar. ``x`` and ``y`` must either have same shape or be
      broadcast compatible.

  Returns:
    An array containing boolean values. ``True`` if the elements of ``x >= y``,
    and ``False`` otherwise.

  See also:
    - :func:`jax.numpy.less_equal`: Returns element-wise truth value of ``x <= y``.
    - :func:`jax.numpy.greater`: Returns element-wise truth value of ``x > y``.
    - :func:`jax.numpy.less`: Returns element-wise truth value of ``x < y``.

  Examples:
    Scalar inputs:

    >>> jnp.greater_equal(4, 7)
    Array(False, dtype=bool, weak_type=True)

    Inputs with same shape:

    >>> x = jnp.array([2, 5, -1])
    >>> y = jnp.array([-6, 4, 3])
    >>> jnp.greater_equal(x, y)
    Array([ True,  True, False], dtype=bool)

    Inputs with broadcast compatibility:

    >>> x1 = jnp.array([[3, -1, 4],
    ...                 [5, 9, -6]])
    >>> y1 = jnp.array([-1, 4, 2])
    >>> jnp.greater_equal(x1, y1)
    Array([[ True, False,  True],
           [ True,  True, False]], dtype=bool)
  """
  return _complex_comparison(lax.ge, *promote_args("greater_equal", x, y))


@export
@partial(jit, inline=True)
def greater(x: ArrayLike, y: ArrayLike, /) -> Array:
  """Return element-wise truth value of ``x > y``.

  JAX implementation of :obj:`numpy.greater`.

  Args:
    x: input array or scalar.
    y: input array or scalar. ``x`` and ``y`` must either have same shape or be
      broadcast compatible.

  Returns:
    An array containing boolean values. ``True`` if the elements of ``x > y``,
    and ``False`` otherwise.

  See also:
    - :func:`jax.numpy.less`: Returns element-wise truth value of ``x < y``.
    - :func:`jax.numpy.greater_equal`: Returns element-wise truth value of
      ``x >= y``.
    - :func:`jax.numpy.less_equal`: Returns element-wise truth value of ``x <= y``.

  Examples:
    Scalar inputs:

    >>> jnp.greater(5, 2)
    Array(True, dtype=bool, weak_type=True)

    Inputs with same shape:

    >>> x = jnp.array([5, 9, -2])
    >>> y = jnp.array([4, -1, 6])
    >>> jnp.greater(x, y)
    Array([ True,  True, False], dtype=bool)

    Inputs with broadcast compatibility:

    >>> x1 = jnp.array([[5, -6, 7],
    ...                 [-2, 5, 9]])
    >>> y1 = jnp.array([-4, 3, 10])
    >>> jnp.greater(x1, y1)
    Array([[ True, False, False],
           [ True,  True, False]], dtype=bool)
  """
  return _complex_comparison(lax.gt, *promote_args("greater", x, y))


@export
@partial(jit, inline=True)
def less_equal(x: ArrayLike, y: ArrayLike, /) -> Array:
  """Return element-wise truth value of ``x <= y``.

  JAX implementation of :obj:`numpy.less_equal`.

  Args:
    x: input array or scalar.
    y: input array or scalar. ``x`` and ``y`` must have either same shape or be
      broadcast compatible.

  Returns:
    An array containing the boolean values. ``True`` if the elements of ``x <= y``,
    and ``False`` otherwise.

  See also:
    - :func:`jax.numpy.greater_equal`: Returns element-wise truth value of
      ``x >= y``.
    - :func:`jax.numpy.greater`: Returns element-wise truth value of ``x > y``.
    - :func:`jax.numpy.less`: Returns element-wise truth value of ``x < y``.

  Examples:
    Scalar inputs:

    >>> jnp.less_equal(6, -2)
    Array(False, dtype=bool, weak_type=True)

    Inputs with same shape:

    >>> x = jnp.array([-4, 1, 7])
    >>> y = jnp.array([2, -3, 8])
    >>> jnp.less_equal(x, y)
    Array([ True, False,  True], dtype=bool)

    Inputs with broadcast compatibility:

    >>> x1 = jnp.array([2, -5, 9])
    >>> y1 = jnp.array([[1, -6, 5],
    ...                 [-2, 4, -6]])
    >>> jnp.less_equal(x1, y1)
    Array([[False, False, False],
           [False,  True, False]], dtype=bool)
  """
  return _complex_comparison(lax.le, *promote_args("less_equal", x, y))


@export
@partial(jit, inline=True)
def less(x: ArrayLike, y: ArrayLike, /) -> Array:
  """Return element-wise truth value of ``x < y``.

  JAX implementation of :obj:`numpy.less`.

  Args:
    x: input array or scalar.
    y: input array or scalar. ``x`` and ``y`` must either have same shape or be
      broadcast compatible.

  Returns:
    An array containing boolean values. ``True`` if the elements of ``x < y``,
    and ``False`` otherwise.

  See also:
    - :func:`jax.numpy.greater`: Returns element-wise truth value of ``x > y``.
    - :func:`jax.numpy.greater_equal`: Returns element-wise truth value of
      ``x >= y``.
    - :func:`jax.numpy.less_equal`: Returns element-wise truth value of ``x <= y``.

  Examples:
    Scalar inputs:

    >>> jnp.less(3, 7)
    Array(True, dtype=bool, weak_type=True)

    Inputs with same shape:

    >>> x = jnp.array([5, 9, -3])
    >>> y = jnp.array([1, 6, 4])
    >>> jnp.less(x, y)
    Array([False, False,  True], dtype=bool)

    Inputs with broadcast compatibility:

    >>> x1 = jnp.array([[2, -4, 6, -8],
    ...                 [-1, 5, -3, 7]])
    >>> y1 = jnp.array([0, 3, -5, 9])
    >>> jnp.less(x1, y1)
    Array([[False,  True, False,  True],
           [ True, False, False,  True]], dtype=bool)
  """
  return _complex_comparison(lax.lt, *promote_args("less", x, y))


# Array API aliases
@export
@partial(jit, inline=True)
def acos(x: ArrayLike, /) -> Array:
  """Alias of :func:`jax.numpy.arccos`"""
  return arccos(*promote_args('acos', x))


@export
@partial(jit, inline=True)
def acosh(x: ArrayLike, /) -> Array:
  """Alias of :func:`jax.numpy.arccosh`"""
  return arccosh(*promote_args('acosh', x))


@export
@partial(jit, inline=True)
def asin(x: ArrayLike, /) -> Array:
  """Alias of :func:`jax.numpy.arcsin`"""
  return arcsin(*promote_args('asin', x))


@export
@partial(jit, inline=True)
def asinh(x: ArrayLike, /) -> Array:
  """Alias of :func:`jax.numpy.arcsinh`"""
  return arcsinh(*promote_args('asinh', x))


@export
@partial(jit, inline=True)
def atan(x: ArrayLike, /) -> Array:
  """Alias of :func:`jax.numpy.arctan`"""
  return arctan(*promote_args('atan', x))


@export
@partial(jit, inline=True)
def atanh(x: ArrayLike, /) -> Array:
  """Alias of :func:`jax.numpy.arctanh`"""
  return arctanh(*promote_args('atanh', x))


@export
@partial(jit, inline=True)
def atan2(x1: ArrayLike, x2: ArrayLike, /) -> Array:
  """Alias of :func:`jax.numpy.arctan2`"""
  return arctan2(*promote_args('atan2', x1, x2))


@export
@jit
def bitwise_count(x: ArrayLike, /) -> Array:
  r"""Counts the number of 1 bits in the binary representation of the absolute value
  of each element of ``x``.

  JAX implementation of :obj:`numpy.bitwise_count`.

  Args:
    x: Input array, only accepts integer subtypes

  Returns:
    An array-like object containing the binary 1 bit counts of the absolute value of
    each element in ``x``, with the same shape as ``x`` of dtype uint8.

  Examples:
    >>> x1 = jnp.array([64, 32, 31, 20])
    >>> # 64 = 0b1000000, 32 = 0b100000, 31 = 0b11111, 20 = 0b10100
    >>> jnp.bitwise_count(x1)
    Array([1, 1, 5, 2], dtype=uint8)

    >>> x2 = jnp.array([-16, -7, 7])
    >>> # |-16| = 0b10000, |-7| = 0b111, 7 = 0b111
    >>> jnp.bitwise_count(x2)
    Array([1, 3, 3], dtype=uint8)

    >>> x3 = jnp.array([[2, -7],[-9, 7]])
    >>> # 2 = 0b10, |-7| = 0b111, |-9| = 0b1001, 7 = 0b111
    >>> jnp.bitwise_count(x3)
    Array([[1, 3],
           [2, 3]], dtype=uint8)
  """
  x, = promote_args_numeric("bitwise_count", x)
  # Following numpy we take the absolute value and return uint8.
  return lax.population_count(abs(x)).astype('uint8')


@export
@partial(jit, inline=True)
def right_shift(x1: ArrayLike, x2: ArrayLike, /) -> Array:
  r"""Right shift the bits of ``x1`` to the amount specified in ``x2``.

  JAX implementation of :obj:`numpy.right_shift`.

  Args:
    x1: Input array, only accepts unsigned integer subtypes
    x2: The amount of bits to shift each element in ``x1`` to the right, only accepts
      integer subtypes

  Returns:
    An array-like object containing the right shifted elements of ``x1`` by the
    amount specified in ``x2``, with the same shape as the broadcasted shape of
    ``x1`` and ``x2``.

  Note:
    If ``x1.shape != x2.shape``, they must be compatible for broadcasting to a
    shared shape, this shared shape will also be the shape of the output. Right shifting
    a scalar x1 by scalar x2 is equivalent to ``x1 // 2**x2``.

  Examples:
    >>> def print_binary(x):
    ...   return [bin(int(val)) for val in x]

    >>> x1 = jnp.array([1, 2, 4, 8])
    >>> print_binary(x1)
    ['0b1', '0b10', '0b100', '0b1000']
    >>> x2 = 1
    >>> result = jnp.right_shift(x1, x2)
    >>> result
    Array([0, 1, 2, 4], dtype=int32)
    >>> print_binary(result)
    ['0b0', '0b1', '0b10', '0b100']

    >>> x1 = 16
    >>> print_binary([x1])
    ['0b10000']
    >>> x2 = jnp.array([1, 2, 3, 4])
    >>> result = jnp.right_shift(x1, x2)
    >>> result
    Array([8, 4, 2, 1], dtype=int32)
    >>> print_binary(result)
    ['0b1000', '0b100', '0b10', '0b1']
  """
  x1, x2 = promote_args_numeric(np.right_shift.__name__, x1, x2)
  lax_fn = lax.shift_right_logical if \
    np.issubdtype(x1.dtype, np.unsignedinteger) else lax.shift_right_arithmetic
  return lax_fn(x1, x2)


@export
@partial(jit, inline=True)
def bitwise_right_shift(x1: ArrayLike, x2: ArrayLike, /) -> Array:
  """Alias of :func:`jax.numpy.right_shift`."""
  return right_shift(x1, x2)


@export
@partial(jit, inline=True)
def absolute(x: ArrayLike, /) -> Array:
  r"""Calculate the absolute value element-wise.

  JAX implementation of :obj:`numpy.absolute`.

  This is the same function as :func:`jax.numpy.abs`.

  Args:
    x: Input array

  Returns:
    An array-like object containing the absolute value of each element in ``x``,
    with the same shape as ``x``. For complex valued input, :math:`a + ib`,
    the absolute value is :math:`\sqrt{a^2+b^2}`.

  Examples:
    >>> x1 = jnp.array([5, -2, 0, 12])
    >>> jnp.absolute(x1)
    Array([ 5,  2,  0, 12], dtype=int32)

    >>> x2 = jnp.array([[ 8, -3, 1],[ 0, 9, -6]])
    >>> jnp.absolute(x2)
    Array([[8, 3, 1],
           [0, 9, 6]], dtype=int32)

    >>> x3 = jnp.array([8 + 15j, 3 - 4j, -5 + 0j])
    >>> jnp.absolute(x3)
    Array([17.,  5.,  5.], dtype=float32)
  """
  check_arraylike('absolute', x)
  dt = dtypes.dtype(x)
  return lax.asarray(x) if dt == np.bool_ or dtypes.issubdtype(dt, np.unsignedinteger) else lax.abs(x)


@export
@partial(jit, inline=True)
def abs(x: ArrayLike, /) -> Array:
  """Alias of :func:`jax.numpy.absolute`."""
  return absolute(x)


@export
@jit
def rint(x: ArrayLike, /) -> Array:
  """Rounds the elements of x to the nearest integer

  JAX implementation of :obj:`numpy.rint`.

  Args:
    x: Input array

  Returns:
    An array-like object containing the rounded elements of ``x``. Always promotes
    to inexact.

  Note:
    If an element of x is exactly half way, e.g. ``0.5`` or ``1.5``, rint will round
    to the nearest even integer.

  Examples:
    >>> x1 = jnp.array([5, 4, 7])
    >>> jnp.rint(x1)
    Array([5., 4., 7.], dtype=float32)

    >>> x2 = jnp.array([-2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5, 4.5])
    >>> jnp.rint(x2)
    Array([-2., -2., -0.,  0.,  2.,  2.,  4.,  4.], dtype=float32)

    >>> x3 = jnp.array([-2.5+3.5j, 4.5-0.5j])
    >>> jnp.rint(x3)
    Array([-2.+4.j,  4.-0.j], dtype=complex64)
  """
  check_arraylike('rint', x)
  dtype = dtypes.dtype(x)
  if dtype == bool or dtypes.issubdtype(dtype, np.integer):
    return lax.convert_element_type(x, dtypes.float_)
  if dtypes.issubdtype(dtype, np.complexfloating):
    return lax.complex(rint(lax.real(x)), rint(lax.imag(x)))
  return lax.round(x, lax.RoundingMethod.TO_NEAREST_EVEN)


@export
@jit
def copysign(x1: ArrayLike, x2: ArrayLike, /) -> Array:
  """Copies the sign of each element in ``x2`` to the corresponding element in ``x1``.

  JAX implementation of :obj:`numpy.copysign`.

  Args:
    x1: Input array
    x2: The array whose elements will be used to determine the sign, must be
      broadcast-compatible with ``x1``

  Returns:
    An array object containing the potentially changed elements of ``x1``, always promotes
    to inexact dtype, and has a shape of ``jnp.broadcast_shapes(x1.shape, x2.shape)``

  Examples:
    >>> x1 = jnp.array([5, 2, 0])
    >>> x2 = -1
    >>> jnp.copysign(x1, x2)
    Array([-5., -2., -0.], dtype=float32)

    >>> x1 = jnp.array([6, 8, 0])
    >>> x2 = 2
    >>> jnp.copysign(x1, x2)
    Array([6., 8., 0.], dtype=float32)

    >>> x1 = jnp.array([2, -3])
    >>> x2 = jnp.array([[1],[-4], [5]])
    >>> jnp.copysign(x1, x2)
    Array([[ 2.,  3.],
           [-2., -3.],
           [ 2.,  3.]], dtype=float32)
  """
  x1, x2 = promote_args_inexact("copysign", x1, x2)
  if dtypes.issubdtype(dtypes.dtype(x1), np.complexfloating):
    raise TypeError("copysign does not support complex-valued inputs")
  return _where(signbit(x2).astype(bool), -lax.abs(x1), lax.abs(x1))


@export
@partial(jit, inline=True)
def true_divide(x1: ArrayLike, x2: ArrayLike, /) -> Array:
  """Calculates the division of x1 by x2 element-wise

  JAX implementation of :func:`numpy.true_divide`.

  Args:
    x1: Input array, the dividend
    x2: Input array, the divisor

  Returns:
    An array containing the elementwise quotients, will always use
    floating point division.

  Examples:
    >>> x1 = jnp.array([3, 4, 5])
    >>> x2 = 2
    >>> jnp.true_divide(x1, x2)
    Array([1.5, 2. , 2.5], dtype=float32)

    >>> x1 = 24
    >>> x2 = jnp.array([3, 4, 6j])
    >>> jnp.true_divide(x1, x2)
    Array([8.+0.j, 6.+0.j, 0.-4.j], dtype=complex64)

    >>> x1 = jnp.array([1j, 9+5j, -4+2j])
    >>> x2 = 3j
    >>> jnp.true_divide(x1, x2)
    Array([0.33333334+0.j       , 1.6666666 -3.j       ,
           0.6666667 +1.3333334j], dtype=complex64)

  See Also:
    :func:`jax.numpy.floor_divide` for integer division
  """
  x1, x2 = promote_args_inexact("true_divide", x1, x2)
  return lax.div(x1, x2)


@export
def divide(x1: ArrayLike, x2: ArrayLike, /) -> Array:
  """Alias of :func:`jax.numpy.true_divide`."""
  return true_divide(x1, x2)


@export
@jit
def floor_divide(x1: ArrayLike, x2: ArrayLike, /) -> Array:
  """Calculates the floor division of x1 by x2 element-wise

  JAX implementation of :obj:`numpy.floor_divide`.

  Args:
    x1: Input array, the dividend
    x2: Input array, the divisor

  Returns:
    An array-like object containing each of the quotients rounded down
    to the nearest integer towards negative infinity. This is equivalent
    to ``x1 // x2`` in Python.

  Note:
    ``x1 // x2`` is equivalent to ``jnp.floor_divide(x1, x2)`` for arrays ``x1``
    and ``x2``

  See Also:
    :func:`jax.numpy.divide` and :func:`jax.numpy.true_divide` for floating point
    division.

  Examples:
    >>> x1 = jnp.array([10, 20, 30])
    >>> x2 = jnp.array([3, 4, 7])
    >>> jnp.floor_divide(x1, x2)
    Array([3, 5, 4], dtype=int32)

    >>> x1 = jnp.array([-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5])
    >>> x2 = 3
    >>> jnp.floor_divide(x1, x2)
    Array([-2, -2, -1, -1, -1,  0,  0,  0,  1,  1,  1], dtype=int32)

    >>> x1 = jnp.array([6, 6, 6], dtype=jnp.int32)
    >>> x2 = jnp.array([2.0, 2.5, 3.0], dtype=jnp.float32)
    >>> jnp.floor_divide(x1, x2)
    Array([3., 2., 2.], dtype=float32)
  """
  x1, x2 = promote_args_numeric("floor_divide", x1, x2)
  dtype = dtypes.dtype(x1)
  if dtypes.issubdtype(dtype, np.unsignedinteger):
    return lax.div(x1, x2)
  elif dtypes.issubdtype(dtype, np.integer):
    quotient = lax.div(x1, x2)
    select = logical_and(lax.sign(x1) != lax.sign(x2), lax.rem(x1, x2) != 0)
    # TODO(mattjj): investigate why subtracting a scalar was causing promotion
    return _where(select, quotient - 1, quotient)
  elif dtypes.issubdtype(dtype, np.complexfloating):
    raise TypeError("floor_divide does not support complex-valued inputs")
  else:
    return _float_divmod(x1, x2)[0]


@export
@jit
def divmod(x1: ArrayLike, x2: ArrayLike, /) -> tuple[Array, Array]:
  """Calculates the integer quotient and remainder of x1 by x2 element-wise

  JAX implementation of :obj:`numpy.divmod`.

  Args:
    x1: Input array, the dividend
    x2: Input array, the divisor

  Returns:
    A tuple of arrays ``(x1 // x2, x1 % x2)``.

  See Also:
    - :func:`jax.numpy.floor_divide`: floor division function
    - :func:`jax.numpy.remainder`: remainder function

  Examples:
    >>> x1 = jnp.array([10, 20, 30])
    >>> x2 = jnp.array([3, 4, 7])
    >>> jnp.divmod(x1, x2)
    (Array([3, 5, 4], dtype=int32), Array([1, 0, 2], dtype=int32))

    >>> x1 = jnp.array([-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5])
    >>> x2 = 3
    >>> jnp.divmod(x1, x2)
    (Array([-2, -2, -1, -1, -1,  0,  0,  0,  1,  1,  1], dtype=int32),
     Array([1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2], dtype=int32))

    >>> x1 = jnp.array([6, 6, 6], dtype=jnp.int32)
    >>> x2 = jnp.array([1.9, 2.5, 3.1], dtype=jnp.float32)
    >>> jnp.divmod(x1, x2)
    (Array([3., 2., 1.], dtype=float32),
     Array([0.30000007, 1.        , 2.9       ], dtype=float32))
  """
  x1, x2 = promote_args_numeric("divmod", x1, x2)
  if dtypes.issubdtype(dtypes.dtype(x1), np.integer):
    return floor_divide(x1, x2), remainder(x1, x2)
  else:
    return _float_divmod(x1, x2)


def _float_divmod(x1: ArrayLike, x2: ArrayLike) -> tuple[Array, Array]:
  # see float_divmod in floatobject.c of CPython
  mod = lax.rem(x1, x2)
  div = lax.div(lax.sub(x1, mod), x2)

  ind = lax.bitwise_and(mod != 0, lax.sign(x2) != lax.sign(mod))
  mod = lax.select(ind, mod + x2, mod)
  div = lax.select(ind, div - _constant_like(div, 1), div)

  return lax.round(div), mod


@export
def power(x1: ArrayLike, x2: ArrayLike, /) -> Array:
  """Calculate element-wise base ``x1`` exponential of ``x2``.

  JAX implementation of :obj:`numpy.power`.

  Args:
    x1: scalar or array. Specifies the bases.
    x2: scalar or array. Specifies the exponent. ``x1`` and ``x2`` should either
      have same shape or be broadcast compatible.

  Returns:
    An array containing the base ``x1`` exponentials of ``x2`` with same dtype
    as input.

  Note:
    - When ``x2`` is a concrete integer scalar, ``jnp.power`` lowers to
      :func:`jax.lax.integer_pow`.
    - When ``x2`` is a traced scalar or an array, ``jnp.power`` lowers to
      :func:`jax.lax.pow`.
    - ``jnp.power`` raises a ``TypeError`` for integer type raised to negative
      integer power.
    - ``jnp.power`` returns ``nan`` for negative value raised to the power of
      non-integer values.

  See also:
    - :func:`jax.lax.pow`: Computes element-wise power, :math:`x^y`.
    - :func:`jax.lax.integer_pow`: Computes element-wise power :math:`x^y`, where
      :math:`y` is a fixed integer.
    - :func:`jax.numpy.float_power`: Computes the first array raised to the power
      of second array, element-wise, by promoting to the inexact dtype.
    - :func:`jax.numpy.pow`: Computes the first array raised to the power of second
      array, element-wise.

  Examples:
    Inputs with scalar integers:

    >>> jnp.power(4, 3)
    Array(64, dtype=int32, weak_type=True)

    Inputs with same shape:

    >>> x1 = jnp.array([2, 4, 5])
    >>> x2 = jnp.array([3, 0.5, 2])
    >>> jnp.power(x1, x2)
    Array([ 8.,  2., 25.], dtype=float32)

    Inputs with broadcast compatibility:

    >>> x3 = jnp.array([-2, 3, 1])
    >>> x4 = jnp.array([[4, 1, 6],
    ...                 [1.3, 3, 5]])
    >>> jnp.power(x3, x4)
    Array([[16.,  3.,  1.],
           [nan, 27.,  1.]], dtype=float32)
  """
  check_arraylike("power", x1, x2)
  check_no_float0s("power", x1, x2)

  # We apply special cases, both for algorithmic and autodiff reasons:
  #  1. for *concrete* integer scalar powers (and arbitrary bases), we use
  #     unrolled binary exponentiation specialized on the exponent, which is
  #     more precise for e.g. x ** 2 when x is a float (algorithmic reason!);
  #  2. for integer bases and integer powers, use unrolled binary exponentiation
  #     where the number of steps is determined by a max bit width of 64
  #     (algorithmic reason!);
  #  3. for integer powers and float/complex bases, we apply the lax primitive
  #     without any promotion of input types because in this case we want the
  #     function to be differentiable wrt its first argument at 0;
  #  3. for other cases, perform jnp dtype promotion on the arguments then apply
  #     lax.pow.

  # Case 1: concrete integer scalar powers:
  if core.is_concrete(x2):
    try:
      x2 = operator.index(x2)  # type: ignore[arg-type]
    except TypeError:
      pass
    else:
      x1, = promote_dtypes_numeric(x1)
      return lax.integer_pow(x1, x2)

  # Handle cases #2 and #3 under a jit:
  return _power(x1, x2)

@export
def pow(x1: ArrayLike, x2: ArrayLike, /) -> Array:
  """Alias of :func:`jax.numpy.power`"""
  return power(x1, x2)

@partial(jit, inline=True)
def _power(x1: ArrayLike, x2: ArrayLike) -> Array:
  x1, x2 = promote_shapes("power", x1, x2)  # not dtypes

  # Case 2: bool/integer result
  x1_, x2_ = promote_args_numeric("power", x1, x2)
  if (dtypes.issubdtype(dtypes.dtype(x1_), np.integer) or
      dtypes.issubdtype(dtypes.dtype(x1_), np.bool_)):
    assert np.iinfo(dtypes.dtype(x1_)).bits <= 64  # _pow_int_int assumes <=64bit
    return _pow_int_int(x1_, x2_)

  # Case 3: float/complex base with integer power (special autodiff behavior)
  d1, d2 = dtypes.dtype(x1), dtypes.dtype(x2)
  if dtypes.issubdtype(d1, np.inexact) and dtypes.issubdtype(d2, np.integer):
    return lax.pow(x1, x2)


  # Case 4: do promotion first
  return lax.pow(x1_, x2_)

# TODO(phawkins): add integer pow support to XLA.
def _pow_int_int(x1, x2):
  # Integer power => use binary exponentiation.
  bits = 6  # Anything more would overflow for any x1 > 1
  zero = _constant_like(x2, 0)
  one = _constant_like(x2, 1)
  # Initialize acc carefully such that pow(0, x2) is zero for x2 != 0
  acc = _where(lax.bitwise_and(lax.eq(x1, zero), lax.ne(x2, zero)), zero, one)
  for _ in range(bits):
    acc = _where(lax.bitwise_and(x2, one), lax.mul(acc, x1), acc)
    x1 = lax.mul(x1, x1)
    x2 = lax.shift_right_logical(x2, one)
  return acc


@binary_ufunc(identity=-np.inf, reduce=reductions._logsumexp)
def logaddexp(x1: ArrayLike, x2: ArrayLike, /) -> Array:
  """Compute ``log(exp(x1) + exp(x2))`` avoiding overflow.

  JAX implementation of :obj:`numpy.logaddexp`

  Args:
    x1: input array
    x2: input array

  Returns:
    array containing the result.

  Examples:

  >>> x1 = jnp.array([1, 2, 3])
  >>> x2 = jnp.array([4, 5, 6])
  >>> result1 = jnp.logaddexp(x1, x2)
  >>> result2 = jnp.log(jnp.exp(x1) + jnp.exp(x2))
  >>> print(jnp.allclose(result1, result2))
  True
  """
  x1, x2 = promote_args_inexact("logaddexp", x1, x2)
  return lax_other.logaddexp(x1, x2)


@binary_ufunc(identity=-np.inf, reduce=reductions._logsumexp2)
def logaddexp2(x1: ArrayLike, x2: ArrayLike, /) -> Array:
  """Logarithm of the sum of exponentials of inputs in base-2 avoiding overflow.

  JAX implementation of :obj:`numpy.logaddexp2`.

  Args:
    x1: input array or scalar.
    x2: input array or scalar. ``x1`` and ``x2`` should either have same shape or
      be broadcast compatible.

  Returns:
    An array containing the result, :math:`log_2(2^{x1}+2^{x2})`, element-wise.

  See also:
    - :func:`jax.numpy.logaddexp`: Computes ``log(exp(x1) + exp(x2))``, element-wise.
    - :func:`jax.numpy.log2`: Calculates the base-2 logarithm of ``x`` element-wise.

  Examples:
    >>> x1 = jnp.array([[3, -1, 4],
    ...                 [8, 5, -2]])
    >>> x2 = jnp.array([2, 3, -5])
    >>> result1 = jnp.logaddexp2(x1, x2)
    >>> result2 = jnp.log2(jnp.exp2(x1) + jnp.exp2(x2))
    >>> jnp.allclose(result1, result2)
    Array(True, dtype=bool)
  """
  x1, x2 = promote_args_inexact("logaddexp2", x1, x2)
  ln2 = float(np.log(2))
  return logaddexp(x1 * ln2, x2 * ln2) / ln2


@export
@partial(jit, inline=True)
def log2(x: ArrayLike, /) -> Array:
  """Calculates the base-2 logarithm of ``x`` element-wise.

  JAX implementation of :obj:`numpy.log2`.

  Args:
    x: Input array

  Returns:
    An array containing the base-2 logarithm of each element in ``x``, promotes
    to inexact dtype.

  Examples:
    >>> x1 = jnp.array([0.25, 0.5, 1, 2, 4, 8])
    >>> jnp.log2(x1)
    Array([-2., -1.,  0.,  1.,  2.,  3.], dtype=float32)
  """
  x, = promote_args_inexact("log2", x)
  if dtypes.issubdtype(x.dtype, np.complexfloating):
    r = lax.log(x)
    re = lax.real(r)
    im = lax.imag(r)
    ln2 = lax.log(_constant_like(re, 2))
    return lax.complex(lax.div(re, ln2), lax.div(im, ln2))
  return lax.div(lax.log(x), lax.log(_constant_like(x, 2)))


@export
@partial(jit, inline=True)
def log10(x: ArrayLike, /) -> Array:
  """Calculates the base-10 logarithm of x element-wise

  JAX implementation of :obj:`numpy.log10`.

  Args:
    x: Input array

  Returns:
    An array containing the base-10 logarithm of each element in ``x``, promotes
    to inexact dtype.

  Examples:
    >>> x1 = jnp.array([0.01, 0.1, 1, 10, 100, 1000])
    >>> with jnp.printoptions(precision=2, suppress=True):
    ...   print(jnp.log10(x1))
    [-2. -1.  0.  1.  2.  3.]
  """
  x, = promote_args_inexact("log10", x)
  if dtypes.issubdtype(x.dtype, np.complexfloating):
    r = lax.log(x)
    re = lax.real(r)
    im = lax.imag(r)
    ln10 = lax.log(_constant_like(re, 10))
    return lax.complex(lax.div(re, ln10), lax.div(im, ln10))
  return lax.div(lax.log(x), lax.log(_constant_like(x, 10)))


@export
@partial(jit, inline=True)
def exp2(x: ArrayLike, /) -> Array:
  """Calculate element-wise base-2 exponential of input.

  JAX implementation of :obj:`numpy.exp2`.

  Args:
    x: input array or scalar

  Returns:
    An array containing the base-2 exponential of each element in ``x``, promotes
    to inexact dtype.

  See also:
    - :func:`jax.numpy.log2`: Calculates base-2 logarithm of each element of input.
    - :func:`jax.numpy.exp`: Calculates exponential of each element of the input.
    - :func:`jax.numpy.expm1`: Calculates :math:`e^x-1` of each element of the
      input.

  Examples:
    ``jnp.exp2`` follows the properties of the exponential such as :math:`2^{a+b}
    = 2^a * 2^b`.

    >>> x1 = jnp.array([2, -4, 3, -1])
    >>> x2 = jnp.array([-1, 3, -2, 3])
    >>> jnp.exp2(x1+x2)
    Array([2. , 0.5, 2. , 4. ], dtype=float32)
    >>> jnp.exp2(x1)*jnp.exp2(x2)
    Array([2. , 0.5, 2. , 4. ], dtype=float32)
  """
  x, = promote_args_inexact("exp2", x)
  return lax.exp2(x)


@export
@jit
def signbit(x: ArrayLike, /) -> Array:
  """Return the sign bit of array elements.

  JAX implementation of :obj:`numpy.signbit`.

  Args:
    x: input array. Complex values are not supported.

  Returns:
    A boolean array of the same shape as ``x``, containing ``True``
    where the sign of ``x`` is negative, and ``False`` otherwise.

  See also:
    - :func:`jax.numpy.sign`: return the mathematical sign of array elements,
      i.e. ``-1``, ``0``, or ``+1``.

  Examples:
    :func:`signbit` on boolean values is always ``False``:

    >>> x = jnp.array([True, False])
    >>> jnp.signbit(x)
    Array([False, False], dtype=bool)

    :func:`signbit` on integer values is equivalent to ``x < 0``:

    >>> x = jnp.array([-2, -1, 0, 1, 2])
    >>> jnp.signbit(x)
    Array([ True,  True, False, False, False], dtype=bool)

    :func:`signbit` on floating point values returns the value of the actual
    sign bit from the float representation, including signed zero:

    >>> x = jnp.array([-1.5, -0.0, 0.0, 1.5])
    >>> jnp.signbit(x)
    Array([ True, True, False, False], dtype=bool)

    This also returns the sign bit for special values such as signed NaN
    and signed infinity:

    >>> x = jnp.array([jnp.nan, -jnp.nan, jnp.inf, -jnp.inf])
    >>> jnp.signbit(x)
    Array([False,  True, False,  True], dtype=bool)
    """
  x, = promote_args("signbit", x)
  dtype = dtypes.dtype(x)
  if dtypes.issubdtype(dtype, np.integer):
    return lax.lt(x, _constant_like(x, 0))
  elif dtypes.issubdtype(dtype, np.bool_):
    return lax.full_like(x, False, dtype=np.bool_)
  elif not dtypes.issubdtype(dtype, np.floating):
    raise ValueError(
        "jax.numpy.signbit is not well defined for %s" % dtype)

  info = dtypes.finfo(dtype)
  if info.bits not in _INT_DTYPES:
    raise NotImplementedError(
        "jax.numpy.signbit only supports 16, 32, and 64-bit types.")
  int_type = _INT_DTYPES[info.bits]
  x = lax.bitcast_convert_type(x, int_type)
  return lax.convert_element_type(x >> (info.nexp + info.nmant), np.bool_)


def _normalize_float(x):
  info = dtypes.finfo(dtypes.dtype(x))
  int_type = _INT_DTYPES[info.bits]
  cond = lax.abs(x) < info.tiny
  x1 = _where(cond, x * _lax_const(x, 1 << info.nmant), x)
  x2 = _where(cond, int_type(-info.nmant), int_type(0))
  return lax.bitcast_convert_type(x1, int_type), x2


@export
@jit
def ldexp(x1: ArrayLike, x2: ArrayLike, /) -> Array:
  """Compute x1 * 2 ** x2

  JAX implementation of :func:`numpy.ldexp`.

  Note that XLA does not provide an ``ldexp`` operation, so this
  is implemneted in JAX via a standard multiplication and
  exponentiation.

  Args:
    x1: real-valued input array.
    x2: integer input array. Must be broadcast-compatible with ``x1``.

  Returns:
    ``x1 * 2 ** x2`` computed element-wise.

  See also:
    - :func:`jax.numpy.frexp`: decompose values into mantissa and exponent.

  Examples:
    >>> x1 = jnp.arange(5.0)
    >>> x2 = 10
    >>> jnp.ldexp(x1, x2)
    Array([   0., 1024., 2048., 3072., 4096.], dtype=float32)

    ``ldexp`` can be used to reconstruct the input to ``frexp``:

    >>> x = jnp.array([2., 3., 5., 11.])
    >>> m, e = jnp.frexp(x)
    >>> m
    Array([0.5   , 0.75  , 0.625 , 0.6875], dtype=float32)
    >>> e
    Array([2, 2, 3, 4], dtype=int32)
    >>> jnp.ldexp(m, e)
    Array([ 2.,  3.,  5., 11.], dtype=float32)
  """
  check_arraylike("ldexp", x1, x2)
  x1_dtype = dtypes.dtype(x1)
  x2_dtype = dtypes.dtype(x2)
  if (dtypes.issubdtype(x1_dtype, np.complexfloating)
      or dtypes.issubdtype(x2_dtype, np.inexact)):
    raise ValueError(f"ldexp not supported for input types {(x1_dtype, x2_dtype)}")
  x1, = promote_args_inexact("ldexp", x1)
  x2 = lax.convert_element_type(x2, dtypes.dtype(x1))
  x = x1 * (2 ** x2)
  return _where(isinf(x1) | (x1 == 0), x1, x)


@export
@jit
def frexp(x: ArrayLike, /) -> tuple[Array, Array]:
  """Split floating point values into mantissa and twos exponent.

  JAX implementation of :func:`numpy.frexp`.

  Args:
    x: real-valued array

  Returns:
    A tuple ``(mantissa, exponent)`` where ``mantissa`` is a floating point
    value between -1 and 1, and ``exponent`` is an integer such that
    ``x == mantissa * 2 ** exponent``.

  See also:
    - :func:`jax.numpy.ldexp`: compute the inverse of ``frexp``.

  Examples:
    Split values into mantissa and exponent:

    >>> x = jnp.array([1., 2., 3., 4., 5.])
    >>> m, e = jnp.frexp(x)
    >>> m
    Array([0.5  , 0.5  , 0.75 , 0.5  , 0.625], dtype=float32)
    >>> e
    Array([1, 2, 2, 3, 3], dtype=int32)

    Reconstruct the original array:

    >>> m * 2 ** e
    Array([1., 2., 3., 4., 5.], dtype=float32)
  """
  check_arraylike("frexp", x)
  x, = promote_dtypes_inexact(x)
  if dtypes.issubdtype(x.dtype, np.complexfloating):
    raise TypeError("frexp does not support complex-valued inputs")

  dtype = dtypes.dtype(x)
  info = dtypes.finfo(dtype)
  mask = (1 << info.nexp) - 1
  bias = 1 - info.minexp

  x1, x2 = _normalize_float(x)
  x2 += ((x1 >> info.nmant) & mask) - bias + 1
  x1 &= ~(mask << info.nmant)
  x1 |= (bias - 1) << info.nmant
  x1 = lax.bitcast_convert_type(x1, dtype)

  cond = isinf(x) | isnan(x) | (x == 0)
  x2 = _where(cond, lax._zeros(x2), x2)
  return _where(cond, x, x1), lax.convert_element_type(x2, np.int32)


@export
@jit
def remainder(x1: ArrayLike, x2: ArrayLike, /) -> Array:
  """Returns element-wise remainder of the division.

  JAX implementation of :obj:`numpy.remainder`.

  Args:
    x1: scalar or array. Specifies the dividend.
    x2: scalar or array. Specifies the divisor. ``x1`` and ``x2`` should either
      have same shape or be broadcast compatible.

  Returns:
    An array containing the remainder of element-wise division of ``x1`` by
    ``x2`` with same sign as the elements of ``x2``.

  Note:
    The result of ``jnp.remainder`` is equivalent to ``x1 - x2 * jnp.floor(x1 / x2)``.

  See also:
    - :func:`jax.numpy.mod`: Returns the element-wise remainder of the division.
    - :func:`jax.numpy.fmod`: Calculates the element-wise floating-point modulo
      operation.
    - :func:`jax.numpy.divmod`: Calculates the integer quotient and remainder of
      ``x1`` by ``x2``, element-wise.

  Examples:
    >>> x1 = jnp.array([[3, -1, 4],
    ...                 [8, 5, -2]])
    >>> x2 = jnp.array([2, 3, -5])
    >>> jnp.remainder(x1, x2)
    Array([[ 1,  2, -1],
           [ 0,  2, -2]], dtype=int32)
    >>> x1 - x2 * jnp.floor(x1 / x2)
    Array([[ 1.,  2., -1.],
           [ 0.,  2., -2.]], dtype=float32)
  """
  x1, x2 = promote_args_numeric("remainder", x1, x2)
  zero = _constant_like(x1, 0)
  if dtypes.issubdtype(x2.dtype, np.integer):
    x2 = _where(x2 == 0, lax._ones(x2), x2)
  trunc_mod = lax.rem(x1, x2)
  trunc_mod_not_zero = lax.ne(trunc_mod, zero)
  do_plus = lax.bitwise_and(
      lax.ne(lax.lt(trunc_mod, zero), lax.lt(x2, zero)), trunc_mod_not_zero)
  return lax.select(do_plus, lax.add(trunc_mod, x2), trunc_mod)


@export
def mod(x1: ArrayLike, x2: ArrayLike, /) -> Array:
  """Alias of :func:`jax.numpy.remainder`"""
  return remainder(x1, x2)


@export
@jit
def fmod(x1: ArrayLike, x2: ArrayLike, /) -> Array:
  """Calculate element-wise floating-point modulo operation.

  JAX implementation of :obj:`numpy.fmod`.

  Args:
    x1: scalar or array. Specifies the dividend.
    x2: scalar or array. Specifies the divisor. ``x1`` and ``x2`` should either
       have same shape or be broadcast compatible.

  Returns:
    An array containing the result of the element-wise floating-point modulo
    operation of ``x1`` and ``x2`` with same sign as the elements of ``x1``.

  Note:
    The result of ``jnp.fmod`` is equivalent to ``x1 - x2 * jnp.fix(x1 / x2)``.

  See also:
    - :func:`jax.numpy.mod` and :func:`jax.numpy.remainder`: Returns the element-wise
      remainder of the division.
    - :func:`jax.numpy.divmod`: Calculates the integer quotient and remainder of
      ``x1`` by ``x2``, element-wise.

  Examples:
    >>> x1 = jnp.array([[3, -1, 4],
    ...                 [8, 5, -2]])
    >>> x2 = jnp.array([2, 3, -5])
    >>> jnp.fmod(x1, x2)
    Array([[ 1, -1,  4],
           [ 0,  2, -2]], dtype=int32)
    >>> x1 - x2 * jnp.fix(x1 / x2)
    Array([[ 1., -1.,  4.],
           [ 0.,  2., -2.]], dtype=float32)
  """
  check_arraylike("fmod", x1, x2)
  if dtypes.issubdtype(dtypes.result_type(x1, x2), np.integer):
    x2 = _where(x2 == 0, lax._ones(x2), x2)
  return lax.rem(*promote_args_numeric("fmod", x1, x2))


@export
@partial(jit, inline=True)
def square(x: ArrayLike, /) -> Array:
  """Calculate element-wise square of the input array.

  JAX implementation of :obj:`numpy.square`.

  Args:
    x: input array or scalar.

  Returns:
    An array containing the square of the elements of ``x``.

  Note:
    ``jnp.square`` is equivalent to computing ``jnp.power(x, 2)``.

  See also:
    - :func:`jax.numpy.sqrt`: Calculates the element-wise non-negative square root
      of the input array.
    - :func:`jax.numpy.power`: Calculates the element-wise base ``x1`` exponential
      of ``x2``.
    - :func:`jax.lax.integer_pow`: Computes element-wise power :math:`x^y`, where
      :math:`y` is a fixed integer.
    - :func:`jax.numpy.float_power`: Computes the first array raised to the power
      of second array, element-wise, by promoting to the inexact dtype.

  Examples:
    >>> x = jnp.array([3, -2, 5.3, 1])
    >>> jnp.square(x)
    Array([ 9.      ,  4.      , 28.090002,  1.      ], dtype=float32)
    >>> jnp.power(x, 2)
    Array([ 9.      ,  4.      , 28.090002,  1.      ], dtype=float32)

    For integer inputs:

    >>> x1 = jnp.array([2, 4, 5, 6])
    >>> jnp.square(x1)
    Array([ 4, 16, 25, 36], dtype=int32)

    For complex-valued inputs:

    >>> x2 = jnp.array([1-3j, -1j, 2])
    >>> jnp.square(x2)
    Array([-8.-6.j, -1.+0.j,  4.+0.j], dtype=complex64)
  """
  check_arraylike("square", x)
  x, = promote_dtypes_numeric(x)
  return lax.square(x)


@export
@partial(jit, inline=True)
def deg2rad(x: ArrayLike, /) -> Array:
  r"""Convert angles from degrees to radians.

  JAX implementation of :obj:`numpy.deg2rad`.

  The angle in degrees is converted to radians by:

  .. math::

     deg2rad(x) = x * \frac{pi}{180}

  Args:
    x: scalar or array. Specifies the angle in degrees.

  Returns:
    An array containing the angles in radians.

  See also:
    - :func:`jax.numpy.rad2deg` and :func:`jax.numpy.degrees`: Converts the angles
      from radians to degrees.
    - :func:`jax.numpy.radians`: Alias of ``deg2rad``.

  Examples:
    >>> x = jnp.array([60, 90, 120, 180])
    >>> jnp.deg2rad(x)
    Array([1.0471976, 1.5707964, 2.0943952, 3.1415927], dtype=float32)
    >>> x * jnp.pi / 180
    Array([1.0471976, 1.5707964, 2.0943952, 3.1415927],      dtype=float32, weak_type=True)
  """
  x, = promote_args_inexact("deg2rad", x)
  return lax.mul(x, _lax_const(x, np.pi / 180))


@export
@partial(jit, inline=True)
def rad2deg(x: ArrayLike, /) -> Array:
  r"""Convert angles from radians to degrees.

  JAX implementation of :obj:`numpy.rad2deg`.

  The angle in radians is converted to degrees by:

  .. math::

     rad2deg(x) = x * \frac{180}{pi}

  Args:
    x: scalar or array. Specifies the angle in radians.

  Returns:
    An array containing the angles in degrees.

  See also:
    - :func:`jax.numpy.deg2rad` and :func:`jax.numpy.radians`: Converts the angles
      from degrees to radians.
    - :func:`jax.numpy.degrees`: Alias of ``rad2deg``.

  Examples:
    >>> pi = jnp.pi
    >>> x = jnp.array([pi/4, pi/2, 2*pi/3])
    >>> jnp.rad2deg(x)
    Array([ 45.     ,  90.     , 120.00001], dtype=float32)
    >>> x * 180 / pi
    Array([ 45.,  90., 120.], dtype=float32)
  """
  x, = promote_args_inexact("rad2deg", x)
  return lax.mul(x, _lax_const(x, 180 / np.pi))


@export
def degrees(x: ArrayLike, /) -> Array:
  """Alias of :func:`jax.numpy.rad2deg`"""
  return rad2deg(x)


@export
def radians(x: ArrayLike, /) -> Array:
  """Alias of :func:`jax.numpy.deg2rad`"""
  return deg2rad(x)


@export
@partial(jit, inline=True)
def conjugate(x: ArrayLike, /) -> Array:
  """Return element-wise complex-conjugate of the input.

  JAX implementation of :obj:`numpy.conjugate`.

  Args:
    x: inpuat array or scalar.

  Returns:
    An array containing the complex-conjugate of ``x``.

  See also:
    - :func:`jax.numpy.real`: Returns the element-wise real part of the complex
      argument.
    - :func:`jax.numpy.imag`: Returns the element-wise imaginary part of the
      complex argument.

  Examples:
    >>> jnp.conjugate(3)
    Array(3, dtype=int32, weak_type=True)
    >>> x = jnp.array([2-1j, 3+5j, 7])
    >>> jnp.conjugate(x)
    Array([2.+1.j, 3.-5.j, 7.-0.j], dtype=complex64)
  """
  check_arraylike("conjugate", x)
  return lax.conj(x) if np.iscomplexobj(x) else lax.asarray(x)


@export
def conj(x: ArrayLike, /) -> Array:
  """Alias of :func:`jax.numpy.conjugate`"""
  return conjugate(x)


@export
@partial(jit, inline=True)
def imag(val: ArrayLike, /) -> Array:
  """Return element-wise imaginary of part of the complex argument.

  JAX implementation of :obj:`numpy.imag`.

  Args:
    val: input array or scalar.

  Returns:
    An array containing the imaginary part of the elements of ``val``.

  See also:
    - :func:`jax.numpy.conjugate` and :func:`jax.numpy.conj`: Returns the element-wise
      complex-conjugate of the input.
    - :func:`jax.numpy.real`: Returns the element-wise real part of the complex
      argument.

  Examples:
    >>> jnp.imag(4)
    Array(0, dtype=int32, weak_type=True)
    >>> jnp.imag(5j)
    Array(5., dtype=float32, weak_type=True)
    >>> x = jnp.array([2+3j, 5-1j, -3])
    >>> jnp.imag(x)
    Array([ 3., -1.,  0.], dtype=float32)
  """
  check_arraylike("imag", val)
  return lax.imag(val) if np.iscomplexobj(val) else lax.full_like(val, 0)


@export
@partial(jit, inline=True)
def real(val: ArrayLike, /) -> Array:
  """Return element-wise real part of the complex argument.

  JAX implementation of :obj:`numpy.real`.

  Args:
    val: input array or scalar.

  Returns:
    An array containing the real part of the elements of ``val``.

  See also:
    - :func:`jax.numpy.conjugate` and :func:`jax.numpy.conj`: Returns the element-wise
      complex-conjugate of the input.
    - :func:`jax.numpy.imag`: Returns the element-wise imaginary part of the
      complex argument.

  Examples:
    >>> jnp.real(5)
    Array(5, dtype=int32, weak_type=True)
    >>> jnp.real(2j)
    Array(0., dtype=float32, weak_type=True)
    >>> x = jnp.array([3-2j, 4+7j, -2j])
    >>> jnp.real(x)
    Array([ 3.,  4., -0.], dtype=float32)
  """
  check_arraylike("real", val)
  return lax.real(val) if np.iscomplexobj(val) else lax.asarray(val)


@export
@jit
def modf(x: ArrayLike, /, out=None) -> tuple[Array, Array]:
  """Return element-wise fractional and integral parts of the input array.

  JAX implementation of :obj:`numpy.modf`.

  Args:
    x: input array or scalar.
    out: Not used by JAX.

  Returns:
    An array containing the fractional and integral parts of the elements of ``x``,
    promoting dtypes inexact.

  See also:
    - :func:`jax.numpy.divmod`: Calculates the integer quotient and remainder of
      ``x1`` by ``x2`` element-wise.

  Examples:
    >>> jnp.modf(4.8)
    (Array(0.8000002, dtype=float32, weak_type=True), Array(4., dtype=float32, weak_type=True))
    >>> x = jnp.array([-3.4, -5.7, 0.6, 1.5, 2.3])
    >>> jnp.modf(x)
    (Array([-0.4000001 , -0.6999998 ,  0.6       ,  0.5       ,  0.29999995],      dtype=float32), Array([-3., -5.,  0.,  1.,  2.], dtype=float32))
  """
  check_arraylike("modf", x)
  x, = promote_dtypes_inexact(x)
  if out is not None:
    raise NotImplementedError("The 'out' argument to jnp.modf is not supported.")
  whole = _where(lax.ge(x, lax._zero(x)), floor(x), ceil(x))
  return x - whole, whole


@export
@partial(jit, inline=True)
def isfinite(x: ArrayLike, /) -> Array:
  """Return a boolean array indicating whether each element of input is finite.

  JAX implementation of :obj:`numpy.isfinite`.

  Args:
    x: input array or scalar.

  Returns:
    A boolean array of same shape as ``x`` containing ``True`` where ``x`` is
    not ``inf``, ``-inf``, or ``NaN``, and ``False`` otherwise.

  See also:
    - :func:`jax.numpy.isinf`: Returns a boolean array indicating whether each
      element of input is either positive or negative infinity.
    - :func:`jax.numpy.isposinf`: Returns a boolean array indicating whether each
      element of input is positive infinity.
    - :func:`jax.numpy.isneginf`: Returns a boolean array indicating whether each
      element of input is negative infinity.
    - :func:`jax.numpy.isnan`: Returns a boolean array indicating whether each
      element of input is not a number (``NaN``).

  Examples:
    >>> x = jnp.array([-1, 3, jnp.inf, jnp.nan])
    >>> jnp.isfinite(x)
    Array([ True,  True, False, False], dtype=bool)
    >>> jnp.isfinite(3-4j)
    Array(True, dtype=bool, weak_type=True)
  """
  check_arraylike("isfinite", x)
  dtype = dtypes.dtype(x)
  if dtypes.issubdtype(dtype, np.floating):
    return lax.is_finite(x)
  elif dtypes.issubdtype(dtype, np.complexfloating):
    return lax.bitwise_and(lax.is_finite(real(x)), lax.is_finite(imag(x)))
  else:
    return lax.full_like(x, True, dtype=np.bool_)


@export
@jit
def isinf(x: ArrayLike, /) -> Array:
  """Return a boolean array indicating whether each element of input is infinite.

  JAX implementation of :obj:`numpy.isinf`.

  Args:
    x: input array or scalar.

  Returns:
    A boolean array of same shape as ``x`` containing ``True`` where ``x`` is
    ``inf`` or ``-inf``, and ``False`` otherwise.

  See also:
    - :func:`jax.numpy.isposinf`: Returns a boolean array indicating whether each
      element of input is positive infinity.
    - :func:`jax.numpy.isneginf`: Returns a boolean array indicating whether each
      element of input is negative infinity.
    - :func:`jax.numpy.isfinite`: Returns a boolean array indicating whether each
      element of input is finite.
    - :func:`jax.numpy.isnan`: Returns a boolean array indicating whether each
      element of input is not a number (``NaN``).

  Examples:
    >>> jnp.isinf(jnp.inf)
    Array(True, dtype=bool)
    >>> x = jnp.array([2+3j, -jnp.inf, 6, jnp.inf, jnp.nan])
    >>> jnp.isinf(x)
    Array([False,  True, False,  True, False], dtype=bool)
  """
  check_arraylike("isinf", x)
  dtype = dtypes.dtype(x)
  if dtypes.issubdtype(dtype, np.floating):
    return lax.eq(lax.abs(x), _constant_like(x, np.inf))
  elif dtypes.issubdtype(dtype, np.complexfloating):
    re = lax.real(x)
    im = lax.imag(x)
    return lax.bitwise_or(lax.eq(lax.abs(re), _constant_like(re, np.inf)),
                          lax.eq(lax.abs(im), _constant_like(im, np.inf)))
  else:
    return lax.full_like(x, False, dtype=np.bool_)


def _isposneginf(infinity: float, x: ArrayLike, out) -> Array:
  if out is not None:
    raise NotImplementedError("The 'out' argument to isneginf/isposinf is not supported.")
  dtype = dtypes.dtype(x)
  if dtypes.issubdtype(dtype, np.floating):
    return lax.eq(x, _constant_like(x, infinity))
  elif dtypes.issubdtype(dtype, np.complexfloating):
    raise ValueError("isposinf/isneginf are not well defined for complex types")
  else:
    return lax.full_like(x, False, dtype=np.bool_)


@export
def isposinf(x, /, out=None):
  """
  Return boolean array indicating whether each element of input is positive infinite.

  JAX implementation of :obj:`numpy.isposinf`.

  Args:
    x: input array or scalar. ``complex`` dtype are not supported.

  Returns:
    A boolean array of same shape as ``x`` containing ``True`` where ``x`` is
    ``inf``, and ``False`` otherwise.

  See also:
    - :func:`jax.numpy.isinf`: Returns a boolean array indicating whether each
      element of input is either positive or negative infinity.
    - :func:`jax.numpy.isneginf`: Returns a boolean array indicating whether each
      element of input is negative infinity.
    - :func:`jax.numpy.isfinite`: Returns a boolean array indicating whether each
      element of input is finite.
    - :func:`jax.numpy.isnan`: Returns a boolean array indicating whether each
      element of input is not a number (``NaN``).

  Examples:
    >>> jnp.isposinf(5)
    Array(False, dtype=bool)
    >>> x = jnp.array([-jnp.inf, 5, jnp.inf, jnp.nan, 1])
    >>> jnp.isposinf(x)
    Array([False, False,  True, False, False], dtype=bool)
  """
  return _isposneginf(np.inf, x, out)


@export
def isneginf(x, /, out=None):
  """
  Return boolean array indicating whether each element of input is negative infinite.

  JAX implementation of :obj:`numpy.isneginf`.

  Args:
    x: input array or scalar. ``complex`` dtype are not supported.

  Returns:
    A boolean array of same shape as ``x`` containing ``True`` where ``x`` is
    ``-inf``, and ``False`` otherwise.

  See also:
    - :func:`jax.numpy.isinf`: Returns a boolean array indicating whether each
      element of input is either positive or negative infinity.
    - :func:`jax.numpy.isposinf`: Returns a boolean array indicating whether each
      element of input is positive infinity.
    - :func:`jax.numpy.isfinite`: Returns a boolean array indicating whether each
      element of input is finite.
    - :func:`jax.numpy.isnan`: Returns a boolean array indicating whether each
      element of input is not a number (``NaN``).

  Examples:
    >>> jnp.isneginf(jnp.inf)
    Array(False, dtype=bool)
    >>> x = jnp.array([-jnp.inf, 5, jnp.inf, jnp.nan, 1])
    >>> jnp.isneginf(x)
    Array([ True, False, False, False, False], dtype=bool)
  """
  return _isposneginf(-np.inf, x, out)


@export
@partial(jit, inline=True)
def isnan(x: ArrayLike, /) -> Array:
  """Returns a boolean array indicating whether each element of input is ``NaN``.

  JAX implementation of :obj:`numpy.isnan`.

  Args:
    x: input array or scalar.

  Returns:
    A boolean array of same shape as ``x`` containing ``True`` where ``x`` is
    not a number (i.e. ``NaN``) and ``False`` otherwise.

  See also:
    - :func:`jax.numpy.isfinite`: Returns a boolean array indicating whether each
      element of input is finite.
    - :func:`jax.numpy.isinf`: Returns a boolean array indicating whether each
      element of input is either positive or negative infinity.
    - :func:`jax.numpy.isposinf`: Returns a boolean array indicating whether each
      element of input is positive infinity.
    - :func:`jax.numpy.isneginf`: Returns a boolean array indicating whether each
      element of input is negative infinity.

  Examples:
    >>> jnp.isnan(6)
    Array(False, dtype=bool, weak_type=True)
    >>> x = jnp.array([2, 1+4j, jnp.inf, jnp.nan])
    >>> jnp.isnan(x)
    Array([False, False, False,  True], dtype=bool)
  """
  check_arraylike("isnan", x)
  return lax.ne(x, x)


@export
@jit
def heaviside(x1: ArrayLike, x2: ArrayLike, /) -> Array:
  r"""Compute the heaviside step function.

  JAX implementation of :obj:`numpy.heaviside`.

  The heaviside step function is defined by:

  .. math::

    \mathrm{heaviside}(x1, x2) = \begin{cases}
      0., & x < 0\\
      x2, & x = 0\\
      1., & x > 0.
    \end{cases}

  Args:
    x1: input array or scalar. ``complex`` dtype are not supported.
    x2: scalar or array. Specifies the return values when ``x1`` is ``0``. ``complex``
      dtype are not supported. ``x1`` and ``x2`` must either have same shape or
      broadcast compatible.

  Returns:
    An array containing the heaviside step function of ``x1``, promoting to
    inexact dtype.

  Examples:
    >>> x1 = jnp.array([[-2, 0, 3],
    ...                 [5, -1, 0],
    ...                 [0, 7, -3]])
    >>> x2 = jnp.array([2, 0.5, 1])
    >>> jnp.heaviside(x1, x2)
    Array([[0. , 0.5, 1. ],
           [1. , 0. , 1. ],
           [2. , 1. , 0. ]], dtype=float32)
    >>> jnp.heaviside(x1, 0.5)
    Array([[0. , 0.5, 1. ],
           [1. , 0. , 0.5],
           [0.5, 1. , 0. ]], dtype=float32)
    >>> jnp.heaviside(-3, x2)
    Array([0., 0., 0.], dtype=float32)
  """
  check_arraylike("heaviside", x1, x2)
  x1, x2 = promote_dtypes_inexact(x1, x2)
  zero = _lax_const(x1, 0)
  return _where(lax.lt(x1, zero), zero,
                _where(lax.gt(x1, zero), _lax_const(x1, 1), x2))


@export
@jit
def hypot(x1: ArrayLike, x2: ArrayLike, /) -> Array:
  r"""
  Return element-wise hypotenuse for the given legs of a right angle triangle.

  JAX implementation of :obj:`numpy.hypot`.

  Args:
    x1: scalar or array. Specifies one of the legs of right angle triangle.
      ``complex`` dtype are not supported.
    x2: scalar or array. Specifies the other leg of right angle triangle.
      ``complex`` dtype are not supported. ``x1`` and ``x2`` must either have
      same shape or be broadcast compatible.

  Returns:
    An array containing the hypotenuse for the given given legs ``x1`` and ``x2``
    of a right angle triangle, promoting to inexact dtype.

  Note:
    ``jnp.hypot`` is a more numerically stable way of computing
    ``jnp.sqrt(x1 ** 2 + x2 **2)``.

  Examples:
    >>> jnp.hypot(3, 4)
    Array(5., dtype=float32, weak_type=True)
    >>> x1 = jnp.array([[3, -2, 5],
    ...                 [9, 1, -4]])
    >>> x2 = jnp.array([-5, 6, 8])
    >>> with jnp.printoptions(precision=3, suppress=True):
    ...   jnp.hypot(x1, x2)
    Array([[ 5.831,  6.325,  9.434],
           [10.296,  6.083,  8.944]], dtype=float32)
  """
  x1, x2 = promote_args_inexact("hypot", x1, x2)

  # TODO(micky774): Promote to ValueError when deprecation is complete
  # (began 2024-4-14).
  if dtypes.issubdtype(x1.dtype, np.complexfloating):
    raise ValueError(
      "jnp.hypot is not well defined for complex-valued inputs. "
      "Please convert to real values first, such as by using abs(x)")
  x1, x2 = lax.abs(x1), lax.abs(x2)
  idx_inf = lax.bitwise_or(isposinf(x1), isposinf(x2))
  x1, x2 = maximum(x1, x2), minimum(x1, x2)
  x = _where(x1 == 0, x1, x1 * lax.sqrt(1 + lax.square(lax.div(x2, _where(x1 == 0, lax._ones(x1), x1)))))
  return _where(idx_inf, _lax_const(x, np.inf), x)


@export
@partial(jit, inline=True)
def reciprocal(x: ArrayLike, /) -> Array:
  """Calculate element-wise reciprocal of the input.

  JAX implementation of :obj:`numpy.reciprocal`.

  The reciprocal is calculated by ``1/x``.

  Args:
    x: input array or scalar.

  Returns:
    An array of same shape as ``x`` containing the reciprocal of each element of
    ``x``.

  Note:
    For integer inputs, ``np.reciprocal`` returns rounded integer output, while
    ``jnp.reciprocal`` promotes integer inputs to floating point.

  Examples:
    >>> jnp.reciprocal(2)
    Array(0.5, dtype=float32, weak_type=True)
    >>> jnp.reciprocal(0.)
    Array(inf, dtype=float32, weak_type=True)
    >>> x = jnp.array([1, 5., 4.])
    >>> jnp.reciprocal(x)
    Array([1.  , 0.2 , 0.25], dtype=float32)
  """
  check_arraylike("reciprocal", x)
  x, = promote_dtypes_inexact(x)
  return lax.integer_pow(x, -1)


@export
@jit
def sinc(x: ArrayLike, /) -> Array:
  r"""Calculate the normalized sinc function.

  JAX implementation of :func:`numpy.sinc`.

  The normalized sinc function is given by

  .. math::
     \mathrm{sinc}(x) = \frac{\sin({\pi x})}{\pi x}

  where ``sinc(0)`` returns the limit value of ``1``. The sinc function is
  smooth and infinitely differentiable.

  Args:
    x : input array; will be promoted to an inexact type.

  Returns:
    An array of the same shape as ``x`` containing the result.

  Examples:
    >>> x = jnp.array([-1, -0.5, 0, 0.5, 1])
    >>> with jnp.printoptions(precision=3, suppress=True):
    ...   jnp.sinc(x)
    Array([-0.   ,  0.637,  1.   ,  0.637, -0.   ], dtype=float32)

    Compare this to the naive approach to computing the function, which is
    undefined at zero:

    >>> with jnp.printoptions(precision=3, suppress=True):
    ...   jnp.sin(jnp.pi * x) / (jnp.pi * x)
    Array([-0.   ,  0.637,    nan,  0.637, -0.   ], dtype=float32)

    JAX defines a custom gradient rule for sinc to allow accurate evaluation
    of the gradient at zero even for higher-order derivatives:

    >>> f = jnp.sinc
    >>> for i in range(1, 6):
    ...   f = jax.grad(f)
    ...   print(f"(d/dx)^{i} f(0.0) = {f(0.0):.2f}")
    ...
    (d/dx)^1 f(0.0) = 0.00
    (d/dx)^2 f(0.0) = -3.29
    (d/dx)^3 f(0.0) = 0.00
    (d/dx)^4 f(0.0) = 19.48
    (d/dx)^5 f(0.0) = 0.00
  """
  check_arraylike("sinc", x)
  x, = promote_dtypes_inexact(x)
  eq_zero = lax.eq(x, _lax_const(x, 0))
  pi_x = lax.mul(_lax_const(x, np.pi), x)
  safe_pi_x = _where(eq_zero, _lax_const(x, 1), pi_x)
  return _where(eq_zero, _sinc_maclaurin(0, pi_x),
                lax.div(lax.sin(safe_pi_x), safe_pi_x))


@partial(custom_jvp, nondiff_argnums=(0,))
def _sinc_maclaurin(k, x):
  # compute the kth derivative of x -> sin(x)/x evaluated at zero (since we
  # compute the monomial term in the jvp rule)
  # TODO(mattjj): see https://github.com/jax-ml/jax/issues/10750
  if k % 2:
    return x * 0
  else:
    return x * 0 + _lax_const(x, (-1) ** (k // 2) / (k + 1))

@_sinc_maclaurin.defjvp
def _sinc_maclaurin_jvp(k, primals, tangents):
  (x,), (t,) = primals, tangents
  return _sinc_maclaurin(k, x), _sinc_maclaurin(k + 1, x) * t
