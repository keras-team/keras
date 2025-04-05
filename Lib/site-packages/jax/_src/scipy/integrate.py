# Copyright 2023 The JAX Authors.
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

from functools import partial

from jax import jit
from jax._src.typing import Array, ArrayLike
import jax.numpy as jnp


@partial(jit, static_argnames=('axis',))
def trapezoid(y: ArrayLike, x: ArrayLike | None = None, dx: ArrayLike = 1.0,
              axis: int = -1) -> Array:
  r"""
  Integrate along the given axis using the composite trapezoidal rule.

  JAX implementation of :func:`scipy.integrate.trapezoid`

  The trapezoidal rule approximates the integral under a curve by summing the
  areas of trapezoids formed between adjacent data points.

  Args:
    y: array of data to integrate.
    x: optional array of sample points corresponding to the ``y`` values. If not
       provided, ``x`` defaults to equally spaced with spacing given by ``dx``.
    dx: The spacing between sample points when `x` is None (default: 1.0).
    axis: The axis along which to integrate (default: -1)

  Returns:
    The definite integral approximated by the trapezoidal rule.

  See also:
    :func:`jax.numpy.trapezoid`: NumPy-style API for trapezoidal integration

  Examples:
    Integrate over a regular grid, with spacing 1.0:

    >>> y = jnp.array([1, 2, 3, 2, 3, 2, 1])
    >>> jax.scipy.integrate.trapezoid(y, dx=1.0)
    Array(13., dtype=float32)

    Integrate over an irregular grid:

    >>> x = jnp.array([0, 2, 5, 7, 10, 15, 20])
    >>> jax.scipy.integrate.trapezoid(y, x)
    Array(43., dtype=float32)

    Approximate :math:`\int_0^{2\pi} \sin^2(x)dx`, which equals :math:`\pi`:

    >>> x = jnp.linspace(0, 2 * jnp.pi, 1000)
    >>> y = jnp.sin(x) ** 2
    >>> result = jax.scipy.integrate.trapezoid(y, x)
    >>> jnp.allclose(result, jnp.pi)
    Array(True, dtype=bool)
  """
  return jnp.trapezoid(y, x, dx, axis)
