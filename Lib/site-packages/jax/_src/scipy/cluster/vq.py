# Copyright 2022 The JAX Authors.
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

import operator

from jax import vmap
import jax.numpy as jnp
from jax._src.numpy.util import check_arraylike, promote_dtypes_inexact
from jax._src.typing import Array, ArrayLike


def vq(obs: ArrayLike, code_book: ArrayLike, check_finite: bool = True) -> tuple[Array, Array]:
  """Assign codes from a code book to a set of observations.

  JAX implementation of :func:`scipy.cluster.vq.vq`.

  Assigns each observation vector in ``obs`` to a code from ``code_book``
  based on the nearest Euclidean distance.

  Args:
    obs: array of observation vectors of shape ``(M, N)``. Each row represents
      a single observation. If ``obs`` is one-dimensional, then each entry is
      treated as a length-1 observation.
    code_book: array of codes with shape ``(K, N)``. Each row represents a single
      code vector. If ``code_book`` is one-dimensional, then each entry is treated
      as a length-1 code.
    check_finite: unused in JAX

  Returns:
    A tuple of arrays ``(code, dist)``

    - ``code`` is an integer array of shape ``(M,)`` containing indices ``0 <= i < K``
      of the closest entry in ``code_book`` for the given entry in ``obs``.
    - ``dist`` is a float array of shape ``(M,)`` containing the euclidean
      distance between each observation and the nearest code.

  Examples:
    >>> obs = jnp.array([[1.1, 2.1, 3.1],
    ...                  [5.9, 4.8, 6.2]])
    >>> code_book = jnp.array([[1., 2., 3.],
    ...                        [2., 3., 4.],
    ...                        [3., 4., 5.],
    ...                        [4., 5., 6.]])
    >>> codes, distances = jax.scipy.cluster.vq.vq(obs, code_book)
    >>> print(codes)
    [0 3]
    >>> print(distances)
    [0.17320499 1.9209373 ]
  """
  del check_finite  # unused
  check_arraylike("scipy.cluster.vq.vq", obs, code_book)
  obs_arr, cb_arr = promote_dtypes_inexact(obs, code_book)
  if obs_arr.ndim != cb_arr.ndim:
      raise ValueError("Observation and code_book should have the same rank")
  if obs_arr.ndim == 1:
      obs_arr, cb_arr = obs_arr[..., None], cb_arr[..., None]
  if obs_arr.ndim != 2:
      raise ValueError("ndim different than 1 or 2 are not supported")
  dist = vmap(lambda ob: jnp.linalg.norm(ob[None] - cb_arr, axis=-1))(obs_arr)
  code = jnp.argmin(dist, axis=-1)
  dist_min = vmap(operator.getitem)(dist, code)
  return code, dist_min
