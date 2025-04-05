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

import functools
import re
import typing


import jax
import jax.numpy as jnp


class Rotation(typing.NamedTuple):
  """Rotation in 3 dimensions.

  JAX implementation of :class:`scipy.spatial.transform.Rotation`.

  Examples:
    Construct an object describing a 90 degree rotation about the z-axis:

    >>> from jax.scipy.spatial.transform import Rotation
    >>> r = Rotation.from_euler('z', 90, degrees=True)

    Convert to a rotation vector:

    >>> r.as_rotvec()
    Array([0.       , 0.       , 1.5707964], dtype=float32)

    Convert to rotation matrix:

    >>> r.as_matrix()
    Array([[ 0.        , -0.99999994,  0.        ],
           [ 0.99999994,  0.        ,  0.        ],
           [ 0.        ,  0.        ,  0.99999994]], dtype=float32)

    Compose with another rotation:

    >>> r2 = Rotation.from_euler('x', 90, degrees=True)
    >>> r3 = r * r2
    >>> r3.as_matrix()
    Array([[0., 0., 1.],
           [1., 0., 0.],
           [0., 1., 0.]], dtype=float32)

    See the scipy :class:`~scipy.spatial.transform.Rotation` documentation for
    further examples of manipulating Rotation objects.
  """
  quat: jax.Array

  @classmethod
  def concatenate(cls, rotations: typing.Sequence):
    """Concatenate a sequence of `Rotation` objects."""
    return cls(jnp.concatenate([rotation.quat for rotation in rotations]))

  @classmethod
  def from_euler(cls, seq: str, angles: jax.Array, degrees: bool = False):
    """Initialize from Euler angles."""
    num_axes = len(seq)
    if num_axes < 1 or num_axes > 3:
      raise ValueError("Expected axis specification to be a non-empty "
                       "string of upto 3 characters, got {}".format(seq))
    intrinsic = (re.match(r'^[XYZ]{1,3}$', seq) is not None)
    extrinsic = (re.match(r'^[xyz]{1,3}$', seq) is not None)
    if not (intrinsic or extrinsic):
      raise ValueError("Expected axes from `seq` to be from ['x', 'y', "
                       "'z'] or ['X', 'Y', 'Z'], got {}".format(seq))
    if any(seq[i] == seq[i+1] for i in range(num_axes - 1)):
      raise ValueError("Expected consecutive axes to be different, "
                       "got {}".format(seq))
    angles = jnp.atleast_1d(angles)
    axes = jnp.array([_elementary_basis_index(x) for x in seq.lower()])
    return cls(_elementary_quat_compose(angles, axes, intrinsic, degrees))

  @classmethod
  def from_matrix(cls, matrix: jax.Array):
    """Initialize from rotation matrix."""
    return cls(_from_matrix(matrix))

  @classmethod
  def from_mrp(cls, mrp: jax.Array):
    """Initialize from Modified Rodrigues Parameters (MRPs)."""
    return cls(_from_mrp(mrp))

  @classmethod
  def from_quat(cls, quat: jax.Array):
    """Initialize from quaternions."""
    return cls(_normalize_quaternion(quat))

  @classmethod
  def from_rotvec(cls, rotvec: jax.Array, degrees: bool = False):
    """Initialize from rotation vectors."""
    return cls(_from_rotvec(rotvec, degrees))

  @classmethod
  def identity(cls, num: int | None = None, dtype=float):
    """Get identity rotation(s)."""
    assert num is None
    quat = jnp.array([0., 0., 0., 1.], dtype=dtype)
    return cls(quat)

  @classmethod
  def random(cls, random_key: jax.Array, num: int | None = None):
    """Generate uniformly distributed rotations."""
    # Need to implement scipy.stats.special_ortho_group for this to work...
    raise NotImplementedError()

  def __getitem__(self, indexer):
    """Extract rotation(s) at given index(es) from object."""
    if self.single:
      raise TypeError("Single rotation is not subscriptable.")
    return Rotation(self.quat[indexer])

  def __len__(self):
    """Number of rotations contained in this object."""
    if self.single:
      raise TypeError('Single rotation has no len().')
    else:
      return self.quat.shape[0]

  def __mul__(self, other) -> Rotation:
    """Compose this rotation with the other."""
    return Rotation.from_quat(_compose_quat(self.quat, other.quat))

  def apply(self, vectors: jax.Array, inverse: bool = False) -> jax.Array:
    """Apply this rotation to one or more vectors."""
    return _apply(self.as_matrix(), vectors, inverse)

  def as_euler(self, seq: str, degrees: bool = False):
    """Represent as Euler angles."""
    if len(seq) != 3:
      raise ValueError(f"Expected 3 axes, got {seq}.")
    intrinsic = (re.match(r'^[XYZ]{1,3}$', seq) is not None)
    extrinsic = (re.match(r'^[xyz]{1,3}$', seq) is not None)
    if not (intrinsic or extrinsic):
      raise ValueError("Expected axes from `seq` to be from "
                       "['x', 'y', 'z'] or ['X', 'Y', 'Z'], "
                       "got {}".format(seq))
    if any(seq[i] == seq[i+1] for i in range(2)):
      raise ValueError("Expected consecutive axes to be different, "
                       "got {}".format(seq))
    axes = jnp.array([_elementary_basis_index(x) for x in seq.lower()])
    with jax.numpy_rank_promotion('allow'):
      return _compute_euler_from_quat(self.quat, axes, extrinsic, degrees)

  def as_matrix(self) -> jax.Array:
    """Represent as rotation matrix."""
    return _as_matrix(self.quat)

  def as_mrp(self) -> jax.Array:
    """Represent as Modified Rodrigues Parameters (MRPs)."""
    return _as_mrp(self.quat)

  def as_rotvec(self, degrees: bool = False) -> jax.Array:
    """Represent as rotation vectors."""
    return _as_rotvec(self.quat, degrees)

  def as_quat(self, canonical: bool=False, scalar_first: bool=False) -> jax.Array:
    """Represent as quaternions."""
    quat = _make_canonical(self.quat) if canonical else self.quat
    if scalar_first:
        return jnp.roll(quat, shift=1, axis=-1)
    return quat

  def inv(self):
    """Invert this rotation."""
    return Rotation(_inv(self.quat))

  def magnitude(self) -> jax.Array:
    """Get the magnitude(s) of the rotation(s)."""
    return _magnitude(self.quat)

  def mean(self, weights: jax.Array | None = None):
    """Get the mean of the rotations."""
    w = jnp.ones(self.quat.shape[0], dtype=self.quat.dtype) if weights is None else jnp.asarray(weights, dtype=self.quat.dtype)
    if w.ndim != 1:
      raise ValueError("Expected `weights` to be 1 dimensional, got "
                       "shape {}.".format(w.shape))
    if w.shape[0] != len(self):
      raise ValueError("Expected `weights` to have number of values "
                       "equal to number of rotations, got "
                       "{} values and {} rotations.".format(w.shape[0], len(self)))
    K = jnp.dot(w[jnp.newaxis, :] * self.quat.T, self.quat)
    _, v = jnp.linalg.eigh(K)
    return Rotation(v[:, -1])

  @property
  def single(self) -> bool:
    """Whether this instance represents a single rotation."""
    return self.quat.ndim == 1


class Slerp(typing.NamedTuple):
  """Spherical Linear Interpolation of Rotations.

  JAX implementation of :class:`scipy.spatial.transform.Slerp`.

  Examples:
    Create a Slerp instance from a series of rotations:

    >>> import math
    >>> from jax.scipy.spatial.transform import Rotation, Slerp
    >>> rots = jnp.array([[90, 0, 0],
    ...                   [0, 45, 0],
    ...                   [0, 0, -30]])
    >>> key_rotations = Rotation.from_euler('zxy', rots, degrees=True)
    >>> key_times = [0, 1, 2]
    >>> slerp = Slerp.init(key_times, key_rotations)
    >>> times = [0, 0.5, 1, 1.5, 2]
    >>> interp_rots = slerp(times)
    >>> interp_rots.as_euler('zxy')
    Array([[ 1.5707963e+00,  0.0000000e+00,  0.0000000e+00],
           [ 8.5309029e-01,  3.8711953e-01,  1.7768645e-01],
           [-2.3841858e-07,  7.8539824e-01,  0.0000000e+00],
           [-5.6668043e-02,  3.9213133e-01, -2.8347540e-01],
           [ 0.0000000e+00,  0.0000000e+00, -5.2359891e-01]], dtype=float32)
  """

  times: jnp.ndarray
  timedelta: jnp.ndarray
  rotations: Rotation
  rotvecs: jnp.ndarray

  @classmethod
  def init(cls, times: jax.Array, rotations: Rotation):
    if not isinstance(rotations, Rotation):
      raise TypeError("`rotations` must be a `Rotation` instance.")
    if rotations.single or len(rotations) == 1:
      raise ValueError("`rotations` must be a sequence of at least 2 rotations.")
    times = jnp.asarray(times, dtype=rotations.quat.dtype)
    if times.ndim != 1:
      raise ValueError("Expected times to be specified in a 1 "
                       "dimensional array, got {} "
                       "dimensions.".format(times.ndim))
    if times.shape[0] != len(rotations):
      raise ValueError("Expected number of rotations to be equal to "
                       "number of timestamps given, got {} rotations "
                       "and {} timestamps.".format(len(rotations), times.shape[0]))
    timedelta = jnp.diff(times)
    # if jnp.any(timedelta <= 0):  # this causes a concretization error...
    #   raise ValueError("Times must be in strictly increasing order.")
    new_rotations = Rotation(rotations.as_quat()[:-1])
    return cls(
      times=times,
      timedelta=timedelta,
      rotations=new_rotations,
      rotvecs=(new_rotations.inv() * Rotation(rotations.as_quat()[1:])).as_rotvec())

  def __call__(self, times: jax.Array):
    """Interpolate rotations."""
    compute_times = jnp.asarray(times, dtype=self.times.dtype)
    if compute_times.ndim > 1:
      raise ValueError("`times` must be at most 1-dimensional.")
    single_time = compute_times.ndim == 0
    compute_times = jnp.atleast_1d(compute_times)
    ind = jnp.maximum(jnp.searchsorted(self.times, compute_times) - 1, 0)
    alpha = (compute_times - self.times[ind]) / self.timedelta[ind]
    result = (self.rotations[ind] * Rotation.from_rotvec(self.rotvecs[ind] * alpha[:, None]))
    if single_time:
      return result[0]
    return result


@functools.partial(jnp.vectorize, signature='(m,m),(m),()->(m)')
def _apply(matrix: jax.Array, vector: jax.Array, inverse: bool) -> jax.Array:
  return jnp.where(inverse, matrix.T, matrix) @ vector


@functools.partial(jnp.vectorize, signature='(m)->(n,n)')
def _as_matrix(quat: jax.Array) -> jax.Array:
  x = quat[0]
  y = quat[1]
  z = quat[2]
  w = quat[3]
  x2 = x * x
  y2 = y * y
  z2 = z * z
  w2 = w * w
  xy = x * y
  zw = z * w
  xz = x * z
  yw = y * w
  yz = y * z
  xw = x * w
  return jnp.array([[+ x2 - y2 - z2 + w2, 2 * (xy - zw), 2 * (xz + yw)],
                    [2 * (xy + zw), - x2 + y2 - z2 + w2, 2 * (yz - xw)],
                    [2 * (xz - yw), 2 * (yz + xw), - x2 - y2 + z2 + w2]])


@functools.partial(jnp.vectorize, signature='(m)->(n)')
def _as_mrp(quat: jax.Array) -> jax.Array:
  sign = jnp.where(quat[3] < 0, -1., 1.)
  denominator = 1. + sign * quat[3]
  return sign * quat[:3] / denominator


@functools.partial(jnp.vectorize, signature='(m),()->(n)')
def _as_rotvec(quat: jax.Array, degrees: bool) -> jax.Array:
  quat = jnp.where(quat[3] < 0, -quat, quat)  # w > 0 to ensure 0 <= angle <= pi
  angle = 2. * jnp.arctan2(_vector_norm(quat[:3]), quat[3])
  angle2 = angle * angle
  small_scale = 2 + angle2 / 12 + 7 * angle2 * angle2 / 2880
  large_scale = angle / jnp.sin(angle / 2)
  scale = jnp.where(angle <= 1e-3, small_scale, large_scale)
  scale = jnp.where(degrees, jnp.rad2deg(scale), scale)
  return scale * jnp.array(quat[:3])


@functools.partial(jnp.vectorize, signature='(n),(n)->(n)')
def _compose_quat(p: jax.Array, q: jax.Array) -> jax.Array:
  cross = jnp.cross(p[:3], q[:3])
  return jnp.array([p[3]*q[0] + q[3]*p[0] + cross[0],
                    p[3]*q[1] + q[3]*p[1] + cross[1],
                    p[3]*q[2] + q[3]*p[2] + cross[2],
                    p[3]*q[3] - p[0]*q[0] - p[1]*q[1] - p[2]*q[2]])

@functools.partial(jnp.vectorize, signature='(m),(l),(),()->(n)')
def _compute_euler_from_quat(quat: jax.Array, axes: jax.Array, extrinsic: bool, degrees: bool) -> jax.Array:
  angle_first = jnp.where(extrinsic, 0, 2)
  angle_third = jnp.where(extrinsic, 2, 0)
  axes = jnp.where(extrinsic, axes, axes[::-1])
  i = axes[0]
  j = axes[1]
  k = axes[2]
  symmetric = i == k
  k = jnp.where(symmetric, 3 - i - j, k)
  sign = jnp.array((i - j) * (j - k) * (k - i) // 2, dtype=quat.dtype)
  eps = 1e-7
  a = jnp.where(symmetric, quat[3], quat[3] - quat[j])
  b = jnp.where(symmetric, quat[i], quat[i] + quat[k] * sign)
  c = jnp.where(symmetric, quat[j], quat[j] + quat[3])
  d = jnp.where(symmetric, quat[k] * sign, quat[k] * sign - quat[i])
  angles = jnp.empty(3, dtype=quat.dtype)
  angles = angles.at[1].set(2 * jnp.arctan2(jnp.hypot(c, d), jnp.hypot(a, b)))
  case = jnp.where(jnp.abs(angles[1] - jnp.pi) <= eps, 2, 0)
  case = jnp.where(jnp.abs(angles[1]) <= eps, 1, case)
  half_sum = jnp.arctan2(b, a)
  half_diff = jnp.arctan2(d, c)
  angles = angles.at[0].set(jnp.where(case == 1, 2 * half_sum, 2 * half_diff * jnp.where(extrinsic, -1, 1)))  # any degenerate case
  angles = angles.at[angle_first].set(jnp.where(case == 0, half_sum - half_diff, angles[angle_first]))
  angles = angles.at[angle_third].set(jnp.where(case == 0, half_sum + half_diff, angles[angle_third]))
  angles = angles.at[angle_third].set(jnp.where(symmetric, angles[angle_third], angles[angle_third] * sign))
  angles = angles.at[1].set(jnp.where(symmetric, angles[1], angles[1] - jnp.pi / 2))
  angles = (angles + jnp.pi) % (2 * jnp.pi) - jnp.pi
  return jnp.where(degrees, jnp.rad2deg(angles), angles)


def _elementary_basis_index(axis: str) -> int:
  if axis == 'x':
    return 0
  elif axis == 'y':
    return 1
  elif axis == 'z':
    return 2
  raise ValueError(f"Expected axis to be from ['x', 'y', 'z'], got {axis}")


@functools.partial(jnp.vectorize, signature=('(m),(m),(),()->(n)'))
def _elementary_quat_compose(angles: jax.Array, axes: jax.Array, intrinsic: bool, degrees: bool) -> jax.Array:
  angles = jnp.where(degrees, jnp.deg2rad(angles), angles)
  result = _make_elementary_quat(axes[0], angles[0])
  for idx in range(1, len(axes)):
    quat = _make_elementary_quat(axes[idx], angles[idx])
    result = jnp.where(intrinsic, _compose_quat(result, quat), _compose_quat(quat, result))
  return result


@functools.partial(jnp.vectorize, signature=('(m),()->(n)'))
def _from_rotvec(rotvec: jax.Array, degrees: bool) -> jax.Array:
  rotvec = jnp.where(degrees, jnp.deg2rad(rotvec), rotvec)
  angle = _vector_norm(rotvec)
  angle2 = angle * angle
  small_scale = scale = 0.5 - angle2 / 48 + angle2 * angle2 / 3840
  large_scale = jnp.sin(angle / 2) / angle
  scale = jnp.where(angle <= 1e-3, small_scale, large_scale)
  return jnp.hstack([scale * rotvec, jnp.cos(angle / 2)])


@functools.partial(jnp.vectorize, signature=('(m,m)->(n)'))
def _from_matrix(matrix: jax.Array) -> jax.Array:
  matrix_trace = matrix[0, 0] + matrix[1, 1] + matrix[2, 2]
  decision = jnp.array([matrix[0, 0], matrix[1, 1], matrix[2, 2], matrix_trace], dtype=matrix.dtype)
  choice = jnp.argmax(decision)
  i = choice
  j = (i + 1) % 3
  k = (j + 1) % 3
  quat_012 = jnp.empty(4, dtype=matrix.dtype)
  quat_012 = quat_012.at[i].set(1 - decision[3] + 2 * matrix[i, i])
  quat_012 = quat_012.at[j].set(matrix[j, i] + matrix[i, j])
  quat_012 = quat_012.at[k].set(matrix[k, i] + matrix[i, k])
  quat_012 = quat_012.at[3].set(matrix[k, j] - matrix[j, k])
  quat_3 = jnp.empty(4, dtype=matrix.dtype)
  quat_3 = quat_3.at[0].set(matrix[2, 1] - matrix[1, 2])
  quat_3 = quat_3.at[1].set(matrix[0, 2] - matrix[2, 0])
  quat_3 = quat_3.at[2].set(matrix[1, 0] - matrix[0, 1])
  quat_3 = quat_3.at[3].set(1 + decision[3])
  quat = jnp.where(choice != 3, quat_012, quat_3)
  return _normalize_quaternion(quat)


@functools.partial(jnp.vectorize, signature='(m)->(n)')
def _from_mrp(mrp: jax.Array) -> jax.Array:
  mrp_squared_plus_1 = jnp.dot(mrp, mrp) + 1
  return jnp.hstack([2 * mrp[:3], (2 - mrp_squared_plus_1)]) / mrp_squared_plus_1


@functools.partial(jnp.vectorize, signature='(n)->(n)')
def _inv(quat: jax.Array) -> jax.Array:
  return quat * jnp.array([-1, -1, -1, 1], dtype=quat.dtype)


@functools.partial(jnp.vectorize, signature='(n)->()')
def _magnitude(quat: jax.Array) -> jax.Array:
  return 2. * jnp.arctan2(_vector_norm(quat[:3]), jnp.abs(quat[3]))


@functools.partial(jnp.vectorize, signature='(),()->(n)')
def _make_elementary_quat(axis: int, angle: jax.Array) -> jax.Array:
  quat = jnp.zeros(4, dtype=angle.dtype)
  quat = quat.at[3].set(jnp.cos(angle / 2.))
  quat = quat.at[axis].set(jnp.sin(angle / 2.))
  return quat


@functools.partial(jnp.vectorize, signature='(n)->(n)')
def _normalize_quaternion(quat: jax.Array) -> jax.Array:
  return quat / _vector_norm(quat)


@functools.partial(jnp.vectorize, signature='(n)->()')
def _vector_norm(vector: jax.Array) -> jax.Array:
  return jnp.sqrt(jnp.dot(vector, vector))


@functools.partial(jnp.vectorize, signature='(n)->(n)')
def _make_canonical(quat: jax.Array) -> jax.Array:
  is_neg = quat < 0
  is_zero = quat == 0

  neg = (
      is_neg[3]
      | (is_zero[3] & is_neg[0])
      | (is_zero[3] & is_zero[0] & is_neg[1])
      | (is_zero[3] & is_zero[0] & is_zero[1] & is_neg[2])
  )

  return jnp.where(neg, -quat, quat)
