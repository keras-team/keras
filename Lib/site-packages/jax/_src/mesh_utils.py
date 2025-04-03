# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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
"""Utils for building a device mesh."""

from __future__ import annotations

import collections
from collections.abc import Callable, Generator, MutableMapping, Sequence
import itertools
import logging
import math
from typing import Any

from jax._src import xla_bridge as xb
import numpy as np

logger = logging.getLogger(__name__)

_TPU_V2 = 'TPU v2'
_TPU_V3 = 'TPU v3'
_TPU_V4 = 'TPU v4'
_TPU_V5_LITE = "TPU v5 lite"
_TPU_V5E = "TPU v5e"
_TPU_V5P = "TPU v5p"

# Maps physical topology -> mesh shape -> transpose to use for jekbradbury's
# famous contiguous mesh trick.
#
# The trick only works for certain topologies and mesh shapes. Trivial dims of
# size 1 can be added to the shapes listed, and they are also supported.
_TRANSPOSE_TRICKS: dict[
    tuple[int, ...], dict[tuple[int, ...], tuple[int, ...]]
] = {
    (2, 2, 1): {
        (2, 2): (0, 1, 2),
    },
    (2, 2, 4): {
        (4, 4): (0, 1, 2),
    },
    (4, 4, 4): {
        (16, 4): (0, 2, 1),
    },
    (4, 8, 8): {
        (64, 4): (0, 2, 1),
        (4, 64): (0, 2, 1),
    },
    (8, 8, 8): {
        (64, 8): (0, 2, 1),
    },
    (8, 16, 16): {
        (256, 8): (0, 2, 1),
        (8, 256): (0, 2, 1),
    },
}

# Physical ordering of core IDs in a tray that creates a ring
_TRAY_RING_ORDER = (0, 1, 2, 3, 6, 7, 4, 5)
_TRAY_2x2_RING_ORDER = (0, 1, 3, 2)
_TRAY_4x4_RING_ORDER = (0, 1, 2, 3, 7, 6, 5, 9, 10, 11, 15, 14, 13, 12, 8, 4)
_V5E_TRAY_RING_ORDER = (0, 1, 2, 3, 7, 6, 5, 4)
_V5E_TRAY_IOTA_ORDER = (0, 4, 2, 6, 1, 5, 3, 7)
_V5P_2x2x2_ORDER = (0, 1, 3, 2, 6, 7, 5, 4)

def _tpu_v2_v3_create_device_mesh(
    mesh_shape: Sequence[int],
    devices: Sequence[Any],
    **unused_kwargs,
) -> np.ndarray:
  if len(devices) == 8:
    logger.info(
        'Reordering mesh to physical ring order on single-tray TPU v2/v3.'
    )
    device_mesh = np.asarray(devices)
    device_mesh = device_mesh[np.array(_TRAY_RING_ORDER)]
    device_mesh = device_mesh.reshape(mesh_shape)
    return device_mesh
  elif mesh_shape[-1] == 8:
    device_mesh = np.asarray(devices).reshape(mesh_shape)
    logger.info(
        'Reordering mesh to physical ring order on each TPU v2/v3 tray.'
    )
    perm = np.array(_TRAY_RING_ORDER)
    device_mesh = device_mesh[..., perm]
    return device_mesh
  else:
    # TODO(skye): implement 2D mesh_shape logic here:
    # https://github.com/tensorflow/lingvo/blob/0df40cf604dfcd14e28f7087d73687a0bd2fe5c6/lingvo/core/gshard_utils.py#L187
    # (possibly replaces above mesh_shape[-1] == 8 case)
    return np.asarray(devices).reshape(mesh_shape)


def _v5e_create_device_mesh(
    mesh_shape: Sequence[int], devices: Sequence[Any], **unused_kwargs
) -> np.ndarray | None:
  """Creates rotated pincer device assignment for selected topologies.

  Args:
    mesh_shape: Logical mesh shape used by the model.
    devices: TPU devices.
    **unused_kwargs: ...

  Returns:
    None or reordered devices reshaped as `mesh_shape`.
  """
  max_x, max_y, max_z = max(getattr(d, "coords", (0, 0, 0)) for d in devices)
  bound_x, bound_y, bound_z = max_x + 1, max_y + 1, max_z + 1
  # Our ring re-ordering makes sense only if the passed-in devices are
  # sequential, which may not always be the case. reversed() changes z-minor to
  # x-minor.
  sequential_devices = sorted(
      devices,
      key=lambda d: tuple(reversed(getattr(d, "coords", (0, 0, 0)))))

  if bound_x == bound_y == 2 and bound_z == 1 and len(devices) == 4:
    device_mesh = np.asarray(sequential_devices)
    device_mesh = device_mesh[np.array(_TRAY_2x2_RING_ORDER)]
    device_mesh = device_mesh.reshape(mesh_shape)
    return device_mesh

  if len(devices) == 8:
    device_mesh = np.asarray(sequential_devices)
    if bound_x == bound_y == bound_z == 2:  # v5e 2x2x2
      order = _V5E_TRAY_IOTA_ORDER
    else:
      order = _V5E_TRAY_RING_ORDER
    device_mesh = device_mesh[np.array(order)]
    device_mesh = device_mesh.reshape(mesh_shape)
    return device_mesh

  if bound_x == bound_y == 4 and bound_z == 1 and len(devices) == 16:  # v5e4x4
    # Only uses ring order if the whole mesh is a replica group.
    if max(mesh_shape) == len(devices):
      device_mesh = np.asarray(sequential_devices)
      device_mesh = device_mesh[np.array(_TRAY_4x4_RING_ORDER)]
      device_mesh = device_mesh.reshape(mesh_shape)
      return device_mesh

  return None


def _v5p_create_device_mesh(
    mesh_shape: Sequence[int], devices: Sequence[Any], **unused_kwargs
) -> np.ndarray | None:
  """Creates device assignment for selected topologies.

  Args:
    mesh_shape: Logical mesh shape used by the model.
    devices: TPU devices.
    **unused_kwargs: ...

  Returns:
    None or reordered devices reshaped as `mesh_shape`.
  """
  max_x, max_y, max_z = max(getattr(d, "coords", (0, 0, 0)) for d in devices)
  bound_x, bound_y, bound_z = max_x + 1, max_y + 1, max_z + 1
  # Our ring re-ordering makes sense only if the passed-in devices are
  # sequential, which may not always be the case. reversed() changes z-minor to
  # x-minor.
  sequential_devices = sorted(
      devices,
      key=lambda d: tuple(reversed(getattr(d, "coords", (0, 0, 0)))))

  if bound_x == bound_y == 2 and bound_z == 2:
    device_mesh = np.asarray(sequential_devices)
    device_mesh = device_mesh[np.array(_V5P_2x2x2_ORDER)]
    device_mesh = device_mesh.reshape(mesh_shape)
    return device_mesh
  return None

# Registers functions to create device mesh for specific device kinds. Takes
# precedence over the more general logic in create_device_mesh(). Handler may
# return None; in that case, it will fall back to using the default logic.
device_kind_handler_dict: dict[
    str,
    Callable[..., np.ndarray | None],
] = {
    _TPU_V2: _tpu_v2_v3_create_device_mesh,
    _TPU_V3: _tpu_v2_v3_create_device_mesh,
    _TPU_V5_LITE: _v5e_create_device_mesh,
    _TPU_V5P: _v5p_create_device_mesh,
}


def _create_device_mesh_for_nd_torus(
    physical_mesh: np.ndarray,
    mesh_shape: Sequence[int],
    *,
    allow_split_physical_axes: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
  """Assigns logical parallelism axes to physical axes of an N-D torus network.

  Given logical parallelism axes with sizes in `mesh_shape` and devices in an
  N-dimensional torus network represented by `physical_mesh`, maps each logical
  axis to one or more physical axes. Prefer to map more-performance-sensitive
  logical axes to larger numbers of physical axes to maximize the bandwidth
  available to them. Also prefer to assign logical axes to multiple physical
  axes of the same size (e.g., a 2D square) rather than multiple physical axes
  of different sizes when possible.

  If allow_split_physical_axes = False (default), this routine will error out
  instead of splitting a physical axis over more than one logical axis (which
  would reduce total usable bandwidth).

  Let's use a concrete example to explain the concepts and considerations.

  As an example, suppose the logical mesh is [data, model], for data and model
  parallelism respectively. Also suppose that data parallelism is less
  performance sensitive than model parallelism. Consider a 3D TPU pod slice of
  shape 4x4x16, represented by a physical mesh of shape (4, 4, 16).

  A TPU pod slice has equal bandwidth along all axes with wraparound links, but
  a 2D plane of size 4x4 may have faster XLA collective implementations than a
  non-square plane or a 1D subgroup. If the mesh_shape is [16, 16], we may want
  the more performance sensitive `model` axis to be mapped to the 4x4 XY plane.

  Args:
    physical_mesh: a np.ndarray of devices in the shape of the N-D torus
      physical topology.
    mesh_shape: shape of the logical mesh (size of the various logical
      parallelism axes), with axes ordered by increasing network intensity.
    allow_split_physical_axes: If True, we would split physical axes if
      necessary to fit the desired mesh shape.

  Returns:
    An np.ndarray of devices in the shape of the logical mesh (mesh_shape), with
      each logical parallelism axis mapped to one or more physical mesh axes.
    The axis assignment matrix, which is a 2-d array mapping from
      (physical_axis, logical_axis) to the size assigned, with the invariant
      np.prod(assignment, axis=1) = physical_mesh_shape, and
      np.prod(assignment, axis=0) = mesh_shape.
  """
  # Remaining physical axes to be assigned to logical axes.
  assignable_physical_mesh = list(physical_mesh.shape)
  # Map each logical axis to a subset of physical axes.
  assignment: list[tuple[int, ...]] = [() for _ in mesh_shape]

  # Assign logical axes from highest network intensity to lowest.
  # `mesh_shape` is assumed to ordered by lowest network intensity first, so
  # reverse it first.
  for logical_axis_index, logical_axis_size in reversed(
      list(enumerate(mesh_shape))
  ):
    # Preferentially map to more physical axes first for higher bandwidth.
    for num_axes in range(3, 0, -1):
      # Try assign to any subset of size num_axes. Generate all candidates.
      indices_and_axes = itertools.combinations(
          enumerate(assignable_physical_mesh), num_axes
      )
      for elem in indices_and_axes:
        c_indices, c_axes = zip(*elem)
        # TODO(zhangqiaorjc): Due to limitations in XLA, 2D collectives only
        # implemented for square 2D plane. Mapping a physical axis to two
        # logical axes might be slower for non-square 2D plane, e.g., map 32 to
        # 4x8 or a single axis. If XLA 2D collectives support non-square plane
        # soon, we can continue to preferentially map to 2D plane in general,
        # otherwise, we should treat non-square 2D plane and 1D submesh equally.
        if np.prod(c_axes) == logical_axis_size:
          assignment[logical_axis_index] = c_indices
          # Zero the assigned physical axes.
          assignable_physical_mesh = [
              0 if i in c_indices else v
              for i, v in enumerate(assignable_physical_mesh)
          ]
          break
      if assignment[logical_axis_index]:
        # We already found an assignment from one candidate above.
        break
    else:
      # If the num_axes for loop did not break, i.e. none of the candidates work
      # goto here with this while-else construct.
      if logical_axis_size > 1:
        if not allow_split_physical_axes:
          # Although this is now implemented, there are downstream tasks
          # counting on this being a NotImplementedError.
          raise NotImplementedError(
              'Failed to find assignment for logical_axis_index'
              f' {logical_axis_index} of size {logical_axis_size} with'
              f' remaining assignable mesh {assignable_physical_mesh}. The size'
              ' of each axis in your logical mesh must be equal to the product'
              ' of some subset of the physical mesh axis sizes. E.g. logical'
              ' mesh (4, 16) is compatible with physical mesh 4x4x4 since 4=4'
              ' and 16=4x4. If you want to split physical axes, set '
              ' allow_split_physical_axes to True.'
          )
        else:
          # We will try finding an assignment, even if that means splitting the
          # physical axes, which requires a more sophisticated implementation.
          return _create_device_mesh_for_nd_torus_splitting_axes(
              physical_mesh, mesh_shape
          )

  # Flatten the assignment, e.g., [(), (2,), (0, 1)] -> (2, 0, 1).
  transpose: list[int] = []
  assignment_array = np.ones(
      [len(physical_mesh.shape), len(mesh_shape)], dtype=np.int64
  )
  for i, x in enumerate(assignment):
    for y in x:
      physical_mesh_axis = int(y)
      assignment_array[physical_mesh_axis, i] = physical_mesh.shape[
          physical_mesh_axis
      ]
      transpose.append(physical_mesh_axis)
  return (
      physical_mesh.transpose(transpose).reshape(mesh_shape),
      assignment_array,
  )


def _create_device_mesh_for_nd_torus_splitting_axes(
    physical_mesh: np.ndarray,
    mesh_shape: Sequence[int],
) -> tuple[np.ndarray, np.ndarray]:
  """Assigns logical parallelism axes to physical axes of an N-D torus network.

  This implementation allows creating meshes that requires splitting physical
  axes, and thus one could produce logical mesh of any shape, as long as the
  number of devices matches, e.g.,

  - Creating 2x2x4 from 4x4;

  - Creating 2x2x16 from 8x8;

  Args:
    physical_mesh: a np.ndarray of devices in the shape of the N-D torus
      physical topology.
    mesh_shape: shape of the logical mesh (size of the various logical
      parallelism axes), with axes ordered by increasing network intensity.

  Returns:
    An np.ndarray of devices in the shape of the logical mesh (mesh_shape), with
      each logical parallelism axis mapped to one or more physical mesh axes.
    The axis assignment matrix, which is a 2-d array mapping from
      (physical_axis, logical_axis) to the size assigned, with the invariant
      np.prod(assignment, axis=1) = physical_mesh_shape, and
      np.prod(assignment, axis=0) = mesh_shape.
  """
  if np.prod(physical_mesh.shape) != np.prod(mesh_shape):
    raise ValueError(
        'The number of devices in physical mesh'
        f' {physical_mesh.shape} does not match the number of devices'
        f' in logical mesh {mesh_shape}.'
    )

  physical_mesh_shape = physical_mesh.shape
  logical_mesh_shape = tuple(mesh_shape)

  # (Partial) assignment map as an 2-d array [p_axis, l_axis] -> size.
  assignment = np.ones(
      [len(physical_mesh_shape), len(logical_mesh_shape)], dtype=np.int64
  )

  # Process logical axes from highest network intensity to lowest.
  # `mesh_shape` is assumed to ordered by lowest network intensity first, so
  # reverse it.
  for logical_axis, logical_axis_size in reversed(
      list(enumerate(logical_mesh_shape))
  ):
    # Go over all the possible assignment for the logical axis, including the
    # one that splits multiple physical axes.
    best_logical_axis_assignment = None
    for logical_axis_assignment in _enumerate_feasible_logical_axis_assignments(
        physical_mesh_shape, assignment, logical_axis_size
    ):
      # TODO(rosun): Instead of using heuristics, replace this with a proper
      # scoring function reflecting the underlying hardware properties.
      if (
          best_logical_axis_assignment is None
          or _prefer_first_logical_axis_assignment(
              logical_axis_assignment,
              best_logical_axis_assignment,
              physical_mesh_shape=physical_mesh_shape,
              assignment=assignment,
          )
      ):
        best_logical_axis_assignment = logical_axis_assignment
    assignment[:, logical_axis] = best_logical_axis_assignment  # type: ignore  # numpy 2.2

  # Read out the assignment.
  logical_mesh = _generate_logical_mesh(
      physical_mesh, logical_mesh_shape, assignment
  )

  return logical_mesh, assignment


def _get_prime_factors(x: int) -> list[int]:
  """Returns a sorted list of prime factors for the given number."""
  assert x > 0
  factors = []
  for p in range(2, math.isqrt(x) + 2):
    while x % p == 0:
      factors.append(p)
      x //= p
    if x == 1:
      return factors
  else:
    return [x]  # x is a prime number.


def _enumerate_feasible_logical_axis_assignments(
    physical_mesh_shape: Sequence[int],
    assignment: np.ndarray,
    logical_axis_size: int,
) -> Generator[np.ndarray, None, None]:
  """Yields feasible assignments for a single logical axis.

  For a physical mesh of shape [x_1, ..., x_n], and the product of all previous
  assignments on each physical axes [y_1, ..., y_n], this function yields all
  possible assignments for the axis as 1-d arrays [z_1, ..., z_n], so that:

  - prod(z_1, ..., z_n) = logical_axis_size

  - x_i % (z_i * y_i) = 0

  Args:
    physical_mesh_shape: Physical mesh shape.
    assignment: Existing assignment matrix.
    logical_axis_size: Size of the logical axis to assign.

  Yields:
    All valid assignments for the logical axis. Each assignment is represented
    as an integer array of length len(physical_mesh_shape).
  """
  logical_axis_factors: MutableMapping[int, int] = collections.defaultdict(int)
  for factor in _get_prime_factors(logical_axis_size):
    logical_axis_factors[factor] += 1

  available_physical_mesh_shape = np.array(physical_mesh_shape) // np.prod(
      assignment, axis=-1
  )

  # To enable efficient enumerations, we first index physical axes by their
  # prime factors. Since we know the prime factorization of the logical axis
  # size, we could simply enumerate by picking the correct count for each
  # prime factor.
  physical_axes_by_factor: MutableMapping[int, list[int]] = (
      collections.defaultdict(list)
  )
  for physical_axis, physical_axis_size in enumerate(
      available_physical_mesh_shape
  ):
    for factor in _get_prime_factors(physical_axis_size):
      if factor not in logical_axis_factors:
        continue
      physical_axes_by_factor[factor].append(physical_axis)

  factors = []
  assignments_by_factor = []
  for factor, multiplicity in logical_axis_factors.items():
    factors.append(factor)
    assignments_by_factor.append(
        set(
            itertools.combinations(
                physical_axes_by_factor[factor], multiplicity
            )
        )
    )

  for axis_assignment in itertools.product(*assignments_by_factor):
    result = np.ones([len(physical_mesh_shape)], dtype=np.int64)
    for factor_index, per_factor_assignment in enumerate(axis_assignment):
      for physical_axis in per_factor_assignment:
        result[physical_axis] *= factors[factor_index]
    yield result


def _prefer_first_logical_axis_assignment(
    x: np.ndarray,
    y: np.ndarray,
    *,
    physical_mesh_shape: Sequence[int],
    assignment: np.ndarray,
) -> bool:
  """Returns True if the first axis assignment is preferred over the second.

  For now, this is implemented with some very simple heuristics. However,
  it is possible to introduce e.g., a value function here based on a more
  precise model of the underlying hardware.

  TODO(rosun): Use a proxy of network capacity to select the partitions.

  Args:
    x: Logical axis assignment as [len(physical_mesh_shape)] array.
    y: Logical axis assignment as [len(physical_mesh_shape)] array.
    physical_mesh_shape: Physical mesh shape.
    assignment: Assignment matrix.

  Returns:
    True if x is preferred over y.
  """
  # Prefer occupying complete physical axes. I don't have a good reason for
  # this, except that it is compatible with the existing behavior.
  #
  # E.g., on 4 x 4 x 8, [4, 4, -] will be preferred over [4, -, 4], and then
  # over [2, 2, 4].
  x_whole_axis_size = np.prod(
      [s for i, s in enumerate(x) if s == physical_mesh_shape[i]]
  )
  y_whole_axis_size = np.prod(
      [s for i, s in enumerate(y) if s == physical_mesh_shape[i]]
  )

  if x_whole_axis_size != y_whole_axis_size:
    return x_whole_axis_size > y_whole_axis_size

  # Prefer occupying more whole physical axes for better bandwidth.
  #
  # This is consistent with existing logic, i.e., 2 x 2 is preferred over 4.
  x_num_whole_axes = len(
      [1 for i, s in enumerate(x) if s == physical_mesh_shape[i] and s > 1]
  )
  y_num_whole_axes = len(
      [1 for i, s in enumerate(y) if s == physical_mesh_shape[i] and s > 1]
  )

  if x_num_whole_axes != y_num_whole_axes:
    return x_num_whole_axes > y_num_whole_axes

  # Prefer taking physical axes that are not taken by logical axes of higher
  # network intensity. E.g., for a 4 x 4 x 4, suppose that the previous
  # assignments are 1 x 2 x 4, and we want to place a new logical axis of size
  # 2, we will go for [2, 1, 1] instead of [1, 2, 1], as the latter choice will
  # tap into bandwidth already taken by the higher intensity axis.
  assigned_physical_mesh_shape = np.prod(assignment, axis=-1)

  x_non_overlapping_axis_size = np.prod(
      [s for i, s in enumerate(x) if assigned_physical_mesh_shape[i] > 1]
  )
  y_non_overlapping_axis_size = np.prod(
      [s for i, s in enumerate(y) if assigned_physical_mesh_shape[i] > 1]
  )

  if x_non_overlapping_axis_size != y_non_overlapping_axis_size:
    return x_non_overlapping_axis_size > y_non_overlapping_axis_size

  # Otherwise sort by reverse lexical graphical order, to be consistent with
  # existing behavior.
  return tuple(x) > tuple(y)


def _generate_logical_mesh(
    physical_mesh: np.ndarray,
    logical_mesh_shape: Sequence[int],
    assignment: np.ndarray,
) -> np.ndarray:
  """Compute the logical mesh from assignment map.

  Args:
    physical_mesh: Physical device mesh.
    logical_mesh_shape: Logical mesh shape.
    assignment: 2-d assignment matrix shape [physical_dims, logical_dims].

  Returns:
    Logical mesh reshaped from physical mesh.
  """
  physical_indices = np.broadcast_to(
      np.expand_dims(
          np.arange(len(physical_mesh.shape), dtype=np.int64), axis=-1
      ),
      assignment.shape,
  ).reshape([-1])

  logical_indices = np.broadcast_to(
      np.expand_dims(
          np.arange(len(logical_mesh_shape), dtype=np.int64), axis=0
      ),
      assignment.shape,
  ).reshape([-1])

  # Axes of logical mesh is ordered by (physical_axis, logical_axis).
  #
  # Note that we sort for each physical_axis the logical_axis, so that higher
  # intensity logical axes are replicated at inner (minor) dimensions.
  #
  # E.g., if a dimension size is 12 = 3x4, where 3 is higher intensity and 4
  # is lower, we want to reshape so that it becomes 12 = 4x3. Imagine in the
  # 1-d case, this will allow more connections between the higher intensity
  # axes.
  logical_mesh = np.reshape(physical_mesh, assignment.reshape([-1]))

  # We will then group by l_axis as this is what is expected from output.
  _, _, transpose_axes = zip(
      *sorted(
          zip(logical_indices, physical_indices, range(len(logical_indices)))
      )
  )
  logical_mesh = np.transpose(logical_mesh, transpose_axes)  # type: ignore  # numpy 2.2

  # Reshape to add the trivial dimensions back.
  logical_mesh = np.reshape(logical_mesh, logical_mesh_shape)  # type: ignore  # numpy 2.2

  return logical_mesh


def _get_physical_tpu_mesh(jax_devices: Sequence[Any]) -> np.ndarray:
  r"""Rearrange TPU devices in a slice into a physical mesh.

  Args:
    jax_devices: A list of JAX devices in a TPU slice in process-tiled z, y, x,
      core order, e.g. from jax.devices(). The coordinates of these devices
      should constitute a cuboid with no holes; e.g., the coordinates can be
      {(1, 0, 0), (1, 0, 1), (1, 1, 0), (1, 1, 1)} (a 1x2x2 cuboid); passing
      only 3 of these devices would result in a "hole" in that cuboid, which is
      an error.  As in our example, the cuboid is not required to include the
      point (0, 0, 0).

  Returns:
    A np.ndarray of JAX devices with shape [global_x, global_y, global_z]. On
      v2 and v3, global_z is instead cores_per_chip (i.e., 2).
  """
  device_kind = jax_devices[0].device_kind
  device_coords = [d.coords for d in jax_devices]
  coord_size = len(device_coords[0])
  # Position-wise max and min coordinates:
  max_coords = tuple(
      max(dc[i] for dc in device_coords) for i in range(coord_size)
  )
  min_coords = tuple(
      min(dc[i] for dc in device_coords) for i in range(coord_size)
  )
  dims = tuple(h - l + 1 for (h, l) in zip(max_coords, min_coords))

  max_cores_per_chip = max(d.core_on_chip for d in jax_devices)
  min_cores_per_chip = min(d.core_on_chip for d in jax_devices)
  cores_per_chip = max_cores_per_chip - min_cores_per_chip + 1

  assert len(dims) == 3, dims
  assert (
      len(jax_devices) == np.prod(dims) * cores_per_chip
  ), f'{jax_devices=} {dims=} {cores_per_chip=}'

  if device_kind in (_TPU_V2, _TPU_V3):
    out = np.empty(dims[:2] + (cores_per_chip,), dtype=object)
    for d in jax_devices:
      coords = d.coords
      assert coords[2] == 0, d
      out[
          coords[0] - min_coords[0],
          coords[1] - min_coords[1],
          d.core_on_chip - min_cores_per_chip,
      ] = d
  else:
    out = np.empty(dims, dtype=object)
    for d in jax_devices:
      coords = d.coords
      if d.core_on_chip != 0:
        raise AssertionError(
            'Creating meshes for TPU >v3 requires one device per chip'
            f' ("megacore" mode). Got device id {d.core_on_chip} for a device'
            f' of kind {device_kind}: {d}.'
        )
      out[
          coords[0] - min_coords[0],
          coords[1] - min_coords[1],
          coords[2] - min_coords[2],
      ] = d

  # Check there is no "hole" in the mesh we constructed.
  if (out == None).any():  # pylint: disable=singleton-comparison
    raise AssertionError(
        'Constructed mesh contains a "hole"; probable cause: coordinates '
        f'of jax_devices are not a contiguous cuboid: {jax_devices}'
    )
  return out


# jekbradbury's famous trick for creating contiguous submeshes (where available)
def _transpose_trick(
    physical_mesh: np.ndarray, mesh_shape: Sequence[int]
) -> np.ndarray:
  mesh_shape = tuple(mesh_shape)
  topology = physical_mesh.shape
  if topology not in _TRANSPOSE_TRICKS:
    raise ValueError(
        'create_device_mesh cannot create contiguous submeshes for '
        f'physical mesh topology {topology}'
    )

  mesh_shape_no_trivial_dims: tuple[int, ...] = ()
  for dim_size in mesh_shape:
    if dim_size != 1:
      mesh_shape_no_trivial_dims += (dim_size,)

  if mesh_shape_no_trivial_dims not in _TRANSPOSE_TRICKS[topology]:
    raise ValueError(
        'create_device_mesh cannot create contiguous submeshes for '
        f'mesh_shape {mesh_shape} and physical mesh topology {topology}. '
        f'Available mesh_shapes: {list(_TRANSPOSE_TRICKS[topology].keys())}'
    )

  return physical_mesh.transpose(
      *_TRANSPOSE_TRICKS[topology][mesh_shape_no_trivial_dims]
  )

def _canonicalize_axis_sizes(axis_sizes: Sequence[int]
                             ) -> tuple[int, ...] | None:
  new_sizes = []
  for s in axis_sizes:
    try:
      new_sizes.append(int(s))
    except:
      return None
  return tuple(new_sizes)

def create_device_mesh(
    mesh_shape: Sequence[int],
    devices: Sequence[Any] | None = None,
    *,
    contiguous_submeshes: bool = False,
    allow_split_physical_axes: bool = False,
) -> np.ndarray:
  """Creates a performant device mesh for jax.sharding.Mesh.

  Args:
    mesh_shape: shape of logical mesh, ordered by increasing network-intensity
      e.g. [replica, data, mdl] where mdl has the most network communication
      requirements.
    devices: optionally, the devices to construct a mesh for. Defaults to
      jax.devices().
    contiguous_submeshes: if True, this function will attempt to create a mesh
      where each process's local devices form a contiguous submesh. A ValueError
      will be raised if this function can't produce a suitable mesh. This
      setting was sometimes necessary before the introduction of jax.Array to
      ensure non-ragged local arrays; if using jax.Arrays, it's better to keep
      this set to False.
    allow_split_physical_axes: If True, we will split physical axes if necessary
      to produce the desired device mesh.

  Raises:
    ValueError: if the number of devices doesn't equal the product of
      `mesh_shape`.

  Returns:
    A np.ndarray of JAX devices with mesh_shape as its shape that can be fed
    into jax.sharding.Mesh with good collective performance.
  """
  if devices is None:
    devices = xb.devices()

  new_mesh_shape = _canonicalize_axis_sizes(mesh_shape)
  if new_mesh_shape is None:
    raise ValueError(
        f'`mesh_shape` passed to `create_device_mesh` should be a sequence of'
        f' ints. Got {mesh_shape}')
  del mesh_shape

  if math.prod(new_mesh_shape) != len(devices):
    raise ValueError(
        f'Number of devices {len(devices)} must equal the product '
        f'of mesh_shape {new_mesh_shape}'
    )
  last_device = devices[-1]

  handler = device_kind_handler_dict.get(last_device.device_kind, None)
  if handler is not None:
    result = handler(
        new_mesh_shape, devices, contiguous_submeshes=contiguous_submeshes
    )
    if result is not None:
      return result

  if last_device.platform == 'tpu':
    physical_mesh = _get_physical_tpu_mesh(devices)
    if contiguous_submeshes:
      physical_mesh = _transpose_trick(physical_mesh, new_mesh_shape)
    device_mesh, _ = _create_device_mesh_for_nd_torus(
        physical_mesh,
        new_mesh_shape,
        allow_split_physical_axes=allow_split_physical_axes,
    )
    return device_mesh
  else:
    device_mesh = np.asarray(devices).reshape(new_mesh_shape)
    return device_mesh


def create_hybrid_device_mesh(
    mesh_shape: Sequence[int],
    dcn_mesh_shape: Sequence[int],
    devices: Sequence[Any] | None = None,
    *,
    process_is_granule: bool = False,
    should_sort_granules_by_key: bool = True,
    allow_split_physical_axes: bool = False,
) -> np.ndarray:
  """Creates a device mesh for hybrid (e.g., ICI and DCN) parallelism.

  Args:
    mesh_shape: shape of the logical mesh for the faster/inner network, ordered
      by increasing network intensity, e.g. [replica, data, mdl] where mdl has
      the most network communication requirements.
    dcn_mesh_shape: shape of the logical mesh for the slower/outer network, in
      the same order as mesh_shape.
    devices: optionally, the devices to construct a mesh for. Defaults to
      jax.devices().
    process_is_granule: if True, this function will treat processes as the units
      of the slower/outer network. Otherwise it will look for slice_index
      attributes on devices and use slices as the units. Enabling this is meant
      as a fallback for platforms that don't set slice_index.
    should_sort_granules_by_key: Whether device granules should be sorted by the
      granule key, either slice or process index, depending on
      process_is_granule.
    allow_split_physical_axes: If True, we will split physical axes if necessary
      to produce the desired device mesh.

  Raises:
    ValueError: if the number of slices to which the `devices` belong doesn't
      equal the product of `dcn_mesh_shape`, or if the number of devices
      belonging to any single slice does not equal the product of `mesh_shape`.

  Returns:
    A np.ndarray of JAX devices with mesh_shape * dcn_mesh_shape as its shape
    that can be fed into jax.sharding.Mesh for hybrid parallelism.
  """
  if devices is None:
    devices = xb.devices()
  attr = 'process_index' if process_is_granule else 'slice_index'
  if not hasattr(devices[0], attr):
    raise ValueError(
        f'Device {devices[0]} does not have attribute {attr}. See'
        ' `process_is_granule` option.'
    )
  granule_dict = collections.defaultdict(list)
  for dev in devices:
    granule_dict[getattr(dev, attr)].append(dev)
  granules = (
      [granule_dict[key] for key in sorted(granule_dict.keys())]
      if should_sort_granules_by_key
      else granule_dict.values()
  )
  if np.prod(dcn_mesh_shape) != len(granules):
    raise ValueError(
        f'Number of slices {len(granules)} must equal the product of '
        f'dcn_mesh_shape {dcn_mesh_shape}'
    )
  per_granule_meshes = [
      create_device_mesh(
          mesh_shape,
          granule,
          allow_split_physical_axes=allow_split_physical_axes,
      )
      for granule in granules
  ]
  # TODO(jekbradbury): handle non-uniform DCN topologies
  granule_mesh = np.arange(len(granules)).reshape(dcn_mesh_shape)
  blocks = np.vectorize(lambda i: per_granule_meshes[i], otypes=[object])(
      granule_mesh
  )
  device_mesh = np.block(blocks.tolist())
  return device_mesh
