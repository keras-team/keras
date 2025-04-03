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
"""Definitions of Mesh and ResourceEnv."""

from __future__ import annotations

import collections
from collections.abc import Hashable, Sequence
import contextlib
import enum
import functools
import math
import threading
from typing import Any, NamedTuple

import numpy as np

from jax._src import config as jax_config
from jax._src import xla_bridge as xb
from jax._src import util
from jax._src.lib import xla_client as xc


MeshAxisName = Any
ResourceAxisName = Hashable


def show_axes(axes):
  return ", ".join(sorted(f"`{a}`" for a in axes))


class ResourceEnv(NamedTuple):
  physical_mesh: Mesh

  def with_mesh(self, mesh: Mesh):
    overlap = set(mesh.axis_names) & (self.resource_axes - set(self.physical_mesh.axis_names))
    if overlap:
      raise ValueError(f"Cannot update the mesh of the current resource "
                       f"environment. The new mesh shadows already defined axes "
                       f"{show_axes(overlap)}")
    return self._replace(physical_mesh=mesh)

  @property
  def physical_resource_axes(self) -> set[ResourceAxisName]:
    return set(self.physical_mesh.axis_names)

  @property
  def resource_axes(self) -> set[ResourceAxisName]:
    return self.physical_resource_axes

  @property
  def shape(self):
    return self.physical_mesh.shape

  @property
  def local_shape(self):
    return self.physical_mesh.local_mesh.shape

  def __repr__(self):
    mesh_repr = ", ".join(
        f"'{k}': {v}" for k, v in self.physical_mesh.shape.items())
    return f"ResourceEnv(mesh=Mesh({mesh_repr}))"


@util.cache(max_size=128, trace_context_in_key=False)
def _get_local_mesh(global_mesh: Mesh, process_index: int) -> Mesh:
  if global_mesh.empty:
    return global_mesh
  is_local_device = np.vectorize(
      lambda d: d.process_index == process_index, otypes=[bool])(global_mesh.devices)
  subcube_indices = []
  # We take the smallest slice of each dimension that doesn't skip any local device.
  for axis in range(global_mesh.devices.ndim):
    other_axes = util.tuple_delete(tuple(range(global_mesh.devices.ndim)), axis)
    # NOTE: This re-reduces over many axes multiple times, so we could definitely
    #       optimize it, but I hope it won't be a bottleneck anytime soon.
    local_slices = is_local_device.any(other_axes, keepdims=False)
    nonzero_indices = np.flatnonzero(local_slices)
    start, end = int(np.min(nonzero_indices)), int(np.max(nonzero_indices))
    subcube_indices.append(slice(start, end + 1))
  subcube_indices_tuple = tuple(subcube_indices)
  # We only end up with all conditions being true if the local devices formed a
  # subcube of the full array. This is because we were biased towards taking a
  # "hull" spanned by the devices, and in case the local devices don't form a
  # subcube that hull will contain non-local devices.
  if not is_local_device[subcube_indices_tuple].all():
    raise ValueError(
        "When passing host local inputs to pjit, devices connected to a single"
        " host must form a contiguous subcube of the global device mesh"
    )
  return Mesh(global_mesh.devices[subcube_indices_tuple], global_mesh.axis_names)


class AxisTypes(enum.Enum):
  Hidden = enum.auto()
  Visible = enum.auto()
  Collective = enum.auto()

  def __repr__(self):
    return self.name

def axis_names_to_types(axis_types) -> dict[str, AxisTypes]:
  return {n: t for t, names in axis_types.items()
          for n in ((names,) if not isinstance(names, tuple) else names)}

def axis_types_to_names(name_to_type: dict[str, AxisTypes]):
  d = collections.defaultdict(list)
  for n, t in name_to_type.items():
    d[t].append(n)
  return {t: ns[0] if len(ns) == 1 else tuple(ns) for t, ns in d.items()}


_mesh_object_dict = {}  # type: ignore

MeshAxisType = dict[AxisTypes, str | tuple[str, ...]]

class Mesh(contextlib.ContextDecorator):
  """Declare the hardware resources available in the scope of this manager.

  In particular, all ``axis_names`` become valid resource names inside the
  managed block and can be used e.g. in the ``in_axis_resources`` argument of
  :py:func:`jax.experimental.pjit.pjit`. Also see JAX's multi-process programming
  model (https://jax.readthedocs.io/en/latest/multi_process.html)
  and the Distributed arrays and automatic parallelization tutorial
  (https://jax.readthedocs.io/en/latest/notebooks/Distributed_arrays_and_automatic_parallelization.html)

  If you are compiling in multiple threads, make sure that the
  ``with Mesh`` context manager is inside the function that the threads will
  execute.

  Args:
    devices: A NumPy ndarray object containing JAX device objects (as
      obtained e.g. from :py:func:`jax.devices`).
    axis_names: A sequence of resource axis names to be assigned to the
      dimensions of the ``devices`` argument. Its length should match the
      rank of ``devices``.

  Examples:

    >>> from jax.experimental.pjit import pjit
    >>> from jax.sharding import Mesh
    >>> from jax.sharding import PartitionSpec as P
    >>> import numpy as np
    ...
    >>> inp = np.arange(16).reshape((8, 2))
    >>> devices = np.array(jax.devices()).reshape(4, 2)
    ...
    >>> # Declare a 2D mesh with axes `x` and `y`.
    >>> global_mesh = Mesh(devices, ('x', 'y'))
    >>> # Use the mesh object directly as a context manager.
    >>> with global_mesh:
    ...   out = pjit(lambda x: x, in_shardings=None, out_shardings=None)(inp)

    >>> # Initialize the Mesh and use the mesh as the context manager.
    >>> with Mesh(devices, ('x', 'y')) as global_mesh:
    ...   out = pjit(lambda x: x, in_shardings=None, out_shardings=None)(inp)

    >>> # Also you can use it as `with ... as ...`.
    >>> global_mesh = Mesh(devices, ('x', 'y'))
    >>> with global_mesh as m:
    ...   out = pjit(lambda x: x, in_shardings=None, out_shardings=None)(inp)

    >>> # You can also use it as `with Mesh(...)`.
    >>> with Mesh(devices, ('x', 'y')):
    ...   out = pjit(lambda x: x, in_shardings=None, out_shardings=None)(inp)
  """

  devices: np.ndarray
  axis_names: tuple[MeshAxisName, ...]
  axis_types: MeshAxisType

  def __new__(cls, devices: np.ndarray | Sequence[xc.Device],
              axis_names: str | Sequence[MeshAxisName], *,
              axis_types: MeshAxisType | None = None):
    if not isinstance(devices, np.ndarray):
      devices = np.array(devices)
    if isinstance(axis_names, str):
      axis_names = (axis_names,)
    axis_names = tuple(axis_names)
    if any(i is None for i in axis_names):
      raise ValueError(f"Mesh axis names cannot be None. Got: {axis_names}")

    if devices.ndim != len(axis_names):
      raise ValueError(
          "Mesh requires the ndim of its first argument (`devices`) to equal "
          "the length of its second argument (`axis_names`), but got "
          f"devices.ndim == {devices.ndim} and "
          f"len(axis_names) == {len(axis_names)}.")

    axis_types = ({AxisTypes.Hidden: axis_names} if axis_types is None else
                  axis_types)
    axis_types_tuple = tuple(axis_types.items())
    if len(axis_names_to_types(axis_types).keys()) != len(axis_names):
      raise ValueError(
          "Number of axis names in axis_types should match the number of"
          f" axis_names. Got axis_names={axis_names} and"
          f" axis_types={axis_types}")

    key = (axis_names, devices.shape, tuple(devices.flat), axis_types_tuple)
    val = _mesh_object_dict.get(key, None)
    if val is not None:
      return val

    self = super().__new__(cls)
    self.devices = devices.copy()
    self.devices.flags.writeable = False
    self.axis_names = axis_names
    self.axis_types = axis_types
    self._axis_types_tuple = axis_types_tuple
    _mesh_object_dict[key] = self
    return self

  def __reduce__(self):
    return (type(self), (self.devices, self.axis_names),
            {'axis_types': self.axis_types})

  def __eq__(self, other):
    if not isinstance(other, Mesh):
      return False
    # This is a performance optimization. Comparing thousands of devices
    # can be expensive.
    if id(self) == id(other):
      return True
    return (self.axis_names == other.axis_names and
            self.devices.shape == other.devices.shape and
            self._axis_types_tuple == other._axis_types_tuple and
            self._internal_device_list == other._internal_device_list)

  def __hash__(self):
    if not hasattr(self, '_hash'):
      self._hash = hash(
          (self.axis_names, self._internal_device_list, self.devices.shape,
           self._axis_types_tuple))
    return self._hash

  def __setattr__(self, name, value):
    if hasattr(self, name):
      if getattr(self, name) == value:
        # This can to happen if two threads race, for example if two threads
        # are trying to hash the same Mesh instance.
        return
      raise RuntimeError(
          f"Cannot reassign attributes ({name}) of immutable mesh objects"
      )
    super().__setattr__(name, value)

  def __enter__(self):
    if jax_config.disallow_mesh_context_manager.value:
      raise RuntimeError("Mesh context manager is disabled.")
    new_env = thread_resources.stack[-1].with_mesh(self)
    thread_resources.stack.append(new_env)
    thread_resources.env = new_env
    jax_config.mesh_context_manager.set_local(
        tuple(t.physical_mesh for t in thread_resources.stack
              if not t.physical_mesh.empty))
    return self

  def __exit__(self, exc_type, exc_value, traceback):
    thread_resources.stack.pop()
    thread_resources.env = thread_resources.stack[-1]
    jax_config.mesh_context_manager.set_local(
        tuple(t.physical_mesh for t in thread_resources.stack
              if not t.physical_mesh.empty))
    return False

  @property
  def shape(self):
    return collections.OrderedDict(
        (name, size)
        for name, size in util.safe_zip(self.axis_names, self.devices.shape))

  @functools.cached_property
  def shape_tuple(self):
    return tuple(
        (name, size)
        for name, size in util.safe_zip(self.axis_names, self.devices.shape))

  @property
  def axis_sizes(self) -> tuple[int, ...]:
    return self.devices.shape

  @functools.cached_property
  def _name_to_type(self):
    return axis_names_to_types(self.axis_types)

  @property
  def size(self):
    return math.prod(self.shape.values()) if self.devices.ndim else 0

  @property
  def empty(self):
    return self.size == 0

  @functools.cached_property
  def is_multi_process(self):
    return self.devices.size != len(self.local_devices)

  @property
  def local_mesh(self):
    return self._local_mesh(xb.process_index())

  def _local_mesh(self, process_index):
    return _get_local_mesh(self, process_index)

  @functools.cached_property
  def device_ids(self):
    assert not self.empty
    return np.vectorize(lambda d: d.id, otypes=[int])(self.devices)

  @functools.cached_property
  def _local_devices_set(self):
    return set(self.local_devices)

  @functools.cached_property
  def _flat_devices_tuple(self):
    return tuple(self.devices.flat)

  @functools.cached_property
  def _internal_device_list(self):
    return xc.DeviceList(self._flat_devices_tuple)

  @functools.cached_property
  def _flat_devices_set(self):
    return set(self.devices.flat)

  def __str__(self):
    mesh_str = ", ".join(f"'{k}': {v}" for k, v in self.shape.items())
    return f"Mesh({mesh_str})"

  @functools.cached_property
  def _repr(self):
    if self.empty:
      return "Mesh(device_ids=[], axis_names=())"
    atr = f", axis_types={self.axis_types}"
    return f"Mesh(device_ids={self.device_ids!r}, axis_names={self.axis_names!r}{atr})"

  def __repr__(self):
    return self._repr

  @functools.cached_property
  def local_devices(self):
    return [d for d in self.devices.flat
            if d.process_index == d.client.process_index()]

  @functools.cached_property
  def abstract_mesh(self):
    return AbstractMesh(self.shape_tuple, axis_types=self.axis_types)

  @functools.cached_property
  def _are_all_axes_collective(self) -> bool:
    return all(t == AxisTypes.Collective for t in self.axis_types.keys())

  @functools.cached_property
  def _are_all_axes_hidden(self) -> bool:
    return all(t == AxisTypes.Hidden for t in self.axis_types.keys())

  @functools.cached_property
  def _any_axis_collective(self) -> bool:
    return any(t == AxisTypes.Collective for t in self.axis_types.keys())

  @functools.cached_property
  def _any_axis_hidden(self) -> bool:
    return any(t == AxisTypes.Hidden for t in self.axis_types.keys())


EMPTY_ENV = ResourceEnv(Mesh(np.empty((), dtype=object), ()))

class _ThreadResourcesLocalState(threading.local):

  def __init__(self):
    self.stack = [EMPTY_ENV]
    self.env = self.stack[-1]

thread_resources = _ThreadResourcesLocalState()


class AbstractMesh:
  """AbstractMesh contains only axis names and axis sizes.

  It does not contain concrete devices compared to `jax.sharding.Mesh`. You
  should use this as an input to the sharding passed to with_sharding_constraint
  and mesh passed to shard_map to avoid tracing and lowering cache misses when
  your mesh shape and axis names stay the same but the devices change.
  See the description of https://github.com/jax-ml/jax/pull/23022 for more
  details.
  """

  def __init__(self, shape_tuple: tuple[tuple[str, int], ...], *,
               axis_types: MeshAxisType | None = None):
    self.shape_tuple = shape_tuple
    if self.shape_tuple:
      self._axis_names, self._axis_sizes = list(zip(*self.shape_tuple))
    else:
      self._axis_names, self._axis_sizes = (), ()
    self.axis_types = ({AxisTypes.Hidden: self._axis_names}
                       if axis_types is None else axis_types)
    self._axis_types_tuple = tuple(self.axis_types.items())
    if len(self._name_to_type.keys()) != len(self._axis_names):
      raise ValueError(
          "Number of axis names in axis_types should match the number of"
          f" axis_names in shape_tuple. Got axis_names={self._axis_names} and"
          f" axis_types={self.axis_types}")

  def __hash__(self):
    return hash((self.shape_tuple, self._axis_types_tuple))

  def __eq__(self, other):
    if not isinstance(other, AbstractMesh):
      return False
    if id(self) == id(other):
      return True
    return (self.shape_tuple == other.shape_tuple and
            self._axis_types_tuple == other._axis_types_tuple)

  def __repr__(self):
    mesh_repr = ", ".join(f"'{n}': {v}" for n, v in self.shape_tuple)
    atr = f", axis_types={self.axis_types}"
    return f"AbstractMesh({mesh_repr}{atr})"

  @property
  def axis_names(self):
    return self._axis_names

  @property
  def axis_sizes(self) -> tuple[int, ...]:
    return self._axis_sizes

  @functools.cached_property
  def _name_to_type(self):
    return axis_names_to_types(self.axis_types)

  @functools.cached_property
  def size(self):
    return math.prod(self._axis_sizes) if self._axis_sizes else 0

  @functools.cached_property
  def shape(self):
    return collections.OrderedDict(self.shape_tuple)

  @property
  def _internal_device_list(self):
    return None

  @property
  def empty(self):
    return self.size == 0

  def update_axis_types(self, new_axis_types) -> AbstractMesh:
    # dict(self._name_to_type) will copy it.
    updated_name_to_type = dict(self._name_to_type)
    updated_name_to_type.update(axis_names_to_types(new_axis_types))
    new_axis_types = axis_types_to_names(updated_name_to_type)
    return AbstractMesh(self.shape_tuple, axis_types=new_axis_types)

  @property
  def abstract_mesh(self):
    return self

  @functools.cached_property
  def _are_all_axes_collective(self) -> bool:
    return all(t == AxisTypes.Collective for t in self.axis_types.keys())

  @functools.cached_property
  def _are_all_axes_hidden(self) -> bool:
    return all(t == AxisTypes.Hidden for t in self.axis_types.keys())

  @functools.cached_property
  def _any_axis_collective(self) -> bool:
    return any(t == AxisTypes.Collective for t in self.axis_types.keys())

  @functools.cached_property
  def _any_axis_hidden(self) -> bool:
    return any(t == AxisTypes.Hidden for t in self.axis_types.keys())

  @property
  def devices(self):
    _raise_value_error("devices")

  @property
  def device_ids(self):
    _raise_value_error("device_ids")

  @property
  def is_multi_process(self):
    _raise_value_error("is_multi_process")

  @property
  def local_devices(self):
    _raise_value_error("local_devices")

  @property
  def local_mesh(self):
    _raise_value_error("local_mesh")

  def __enter__(self):
    _raise_value_error("__enter__")

  def __exit__(self, exc_type, exc_value, traceback):
    _raise_value_error("__exit__")

  @staticmethod
  def _extremely_unsafe_enter_tracing_context(mesh: AbstractMesh):
    jax_config.abstract_mesh_context_manager.set_local(mesh)
    return


# Create this indirection because pytype fails to recognize a property if a
# property raises an exception unconditionally. Remove this once that is fixed.
def _raise_value_error(name):
  raise ValueError(f"AbstractMesh does not implement {name}")


@contextlib.contextmanager
def set_abstract_mesh(mesh: AbstractMesh):
  prev_val = jax_config.abstract_mesh_context_manager.swap_local(mesh)
  try:
    yield
  finally:
    jax_config.abstract_mesh_context_manager.set_local(prev_val)

def get_abstract_mesh():
  return jax_config.abstract_mesh_context_manager.value


@contextlib.contextmanager
def set_concrete_mesh(mesh: Mesh):
  prev_val = jax_config.device_context.swap_local(mesh)
  try:
    yield
  finally:
    jax_config.device_context.set_local(prev_val)

def get_concrete_mesh():
  return jax_config.device_context.value


@contextlib.contextmanager
def use_mesh(mesh: Mesh):
  with (set_abstract_mesh(mesh.abstract_mesh),
        jax_config.sharding_in_types(True), set_concrete_mesh(mesh)):
    yield
