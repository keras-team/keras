# Copyright 2024 The etils Authors.
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

"""Common geometric utils."""

from __future__ import annotations

from etils.enp import checking
from etils.enp import compat
from etils.enp import numpy_utils
from etils.enp.typing import FloatArray


@checking.check_and_normalize_arrays(strict=False)
def batch_dot(
    x0: FloatArray['... n'],
    x1: FloatArray['... n'],
    *,
    keepdims: bool = False,
    xnp: numpy_utils.NpModule = ...,
) -> FloatArray['... 1?']:
  """Dot product on the last dimension, with broadcasting support.

  Contrary to `np.dot`, the behavior is consistent for 1-dim vs n-dim (while
  dot act as matmul).
  First dimensions are always broadcasted.

  Args:
    x0: Vector array
    x1: Vector array
    keepdims: If True, returns `FloatArray['... 1']`
    xnp: Numpy module to use

  Returns:
    The dot product along the last axis.
  """
  # Weirdly, this doesn't seem np has a native ops for this:
  # * `np.dot`: 1-D vs 2-D behave differently
  # * `np.matmul`: Different op (`kj,jn` vs `...k,...k`)
  # * `np.tensordot`: Weird broadcasting
  # * `np.inner`: Weird broadcasting
  y = xnp.einsum('...m,...m->...', x0, x1)
  return y[..., None] if keepdims else y


@checking.check_and_normalize_arrays(strict=False)
def angle_between(
    x0: FloatArray[..., 3],
    x1: FloatArray[..., 3],
    *,
    keepdims: bool = False,
    xnp: numpy_utils.NpModule = ...,
) -> FloatArray['... 1?']:
  """Compute angle between 2 vectors, unsigned."""
  a0 = compat.norm(xnp.cross(x0, x1), axis=-1, keepdims=keepdims)
  a1 = batch_dot(x0, x1, keepdims=keepdims)
  angle = xnp.arctan2(a0, a1)
  return angle


@checking.check_and_normalize_arrays(strict=False)
def project_onto_vector(
    u: FloatArray[..., 3],
    v: FloatArray[..., 3],
) -> FloatArray[..., 3]:
  """Project `u` onto `v`."""
  return (
      batch_dot(u, v, keepdims=True)
      / compat.norm(v, axis=-1, keepdims=True) ** 2
      * v
  )


@checking.check_and_normalize_arrays(strict=False)
def project_onto_plane(
    u: FloatArray[..., 3],
    n: FloatArray[..., 3],
) -> FloatArray[..., 3]:
  """Project `u` onto the plane `n` (orthogonal vector)."""
  return u - project_onto_vector(u, n)
