# Copyright 2021 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Utilities to ensure the implementation is safe wrt numerical issues.

Note that complex numbers are also supported, see
https://gist.github.com/wdphy16/118aef6fb5f82c49790d7678cf87da29
"""

from typing import Optional, Union

import chex
import jax.numpy as jnp
import numpy as np


# TODO(jscholz) Promote these functions to jax core lib?


def abs_sq(x: chex.Array) -> chex.Array:
  """Returns the squared absolute value of a (maybe complex) array.

  For real `x`, JAX generates the same HLO from this, `jnp.square(x)`, `x * x`,
  or `x**2`.

  Args:
    x: a (maybe complex) array.

  Returns:
    The squared absolute value of `x`.
  """
  if not isinstance(x, (np.ndarray, jnp.ndarray)):
    raise ValueError(f'`abs_sq` accepts only NDarrays, got: {x}.')
  return (x.conj() * x).real


def safe_norm(
    x: chex.Array,
    min_norm: chex.Numeric,
    ord: Optional[Union[int, float, str]] = None,  # pylint: disable=redefined-builtin
    axis: Union[None, tuple[int, ...], int] = None,
    keepdims: bool = False,
) -> chex.Array:
  """Returns jnp.maximum(jnp.linalg.norm(x), min_norm) with correct gradients.

  The gradients of `jnp.maximum(jnp.linalg.norm(x), min_norm)` at 0.0 is `NaN`,
  because jax will evaluate both branches of the `jnp.maximum`. This function
  will instead return the correct gradient of 0.0 also in such setting.

  Args:
    x: jax array.
    min_norm: lower bound for the returned norm.
    ord: {non-zero int, inf, -inf, 'fro', 'nuc'}, optional. Order of the norm.
      inf means numpy's inf object. The default is None.
    axis: {None, int, 2-tuple of ints}, optional. If axis is an integer, it
      specifies the axis of x along which to compute the vector norms. If axis
      is a 2-tuple, it specifies the axes that hold 2-D matrices, and the matrix
      norms of these matrices are computed. If axis is None then either a vector
      norm (when x is 1-D) or a matrix norm (when x is 2-D) is returned. The
      default is None.
    keepdims: bool, optional. If this is set to True, the axes which are normed
      over are left in the result as dimensions with size one. With this option
      the result will broadcast correctly against the original x.

  Returns:
    The safe norm of the input vector, accounting for correct gradient.
  """
  norm = jnp.linalg.norm(x, ord=ord, axis=axis, keepdims=True)
  x = jnp.where(norm <= min_norm, jnp.ones_like(x), x)
  norm = jnp.squeeze(norm, axis=axis) if not keepdims else norm
  masked_norm = jnp.linalg.norm(x, ord=ord, axis=axis, keepdims=keepdims)
  return jnp.where(norm <= min_norm, min_norm, masked_norm)


def safe_root_mean_squares(x: chex.Array, min_rms: chex.Numeric) -> chex.Array:
  """Returns `maximum(sqrt(mean(abs_sq(x))), min_norm)` with correct grads.

  The gradients of `maximum(sqrt(mean(abs_sq(x))), min_norm)` at 0.0
  is `NaN`, because jax will evaluate both branches of the `jnp.maximum`. This
  function will instead return the correct gradient of 0.0 also in such setting.

  Args:
    x: jax array.
    min_rms: lower bound for the returned norm.

  Returns:
    The safe RMS of the input vector, accounting for correct gradient.
  """
  rms = jnp.sqrt(jnp.mean(abs_sq(x)))
  x = jnp.where(rms <= min_rms, jnp.ones_like(x), x)
  return jnp.where(rms <= min_rms, min_rms, jnp.sqrt(jnp.mean(abs_sq(x))))


def safe_increment(count: chex.Numeric) -> chex.Numeric:
  """Increments counter by one while avoiding overflow.

  Denote ``max_val``, ``min_val`` as the maximum, minimum, possible values for
  the ``dtype`` of ``count``. Normally ``max_val + 1`` would overflow to
  ``min_val``. This functions ensures that when ``max_val`` is reached the
  counter stays at ``max_val``.

  Args:
    count: a counter to be incremented.

  Returns:
    A counter incremented by 1, or ``max_val`` if the maximum value is
    reached.

  Examples:
    >>> import jax.numpy as jnp
    >>> import optax
    >>> optax.safe_increment(jnp.asarray(1, dtype=jnp.int32))
    Array(2, dtype=int32)
    >>> optax.safe_increment(jnp.asarray(2147483647, dtype=jnp.int32))
    Array(2147483647, dtype=int32)

  .. versionadded:: 0.2.4
  """
  count_dtype = jnp.asarray(count).dtype
  if jnp.issubdtype(count_dtype, jnp.integer):
    max_value = jnp.iinfo(count_dtype).max
  elif jnp.issubdtype(count_dtype, jnp.floating):
    max_value = jnp.finfo(count_dtype).max
  else:
    raise ValueError(
        f'Cannot safely increment count with dtype {count_dtype},'
        ' valid dtypes are subdtypes of "jnp.integer" or "jnp.floating".'
    )
  max_value = jnp.array(max_value, count_dtype)
  one = jnp.array(1, count_dtype)
  return jnp.where(count < max_value, count + one, max_value)


safe_int32_increment = safe_increment
