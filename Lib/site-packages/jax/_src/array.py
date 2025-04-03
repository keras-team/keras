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

from collections import defaultdict
from collections.abc import Callable, Sequence
import enum
import functools
import math
import operator as op
from typing import Any, TYPE_CHECKING, cast

from jax._src import api
from jax._src import basearray
from jax._src import config
from jax._src import core
from jax._src import deprecations
from jax._src import dispatch
from jax._src import dtypes
from jax._src import errors
from jax._src import profiler
from jax._src import util
from jax._src import xla_bridge
from jax._src.interpreters import mlir
from jax._src.interpreters import pxla
from jax._src.interpreters import xla
from jax._src.layout import AutoLayout, DeviceLocalLayout, Layout
from jax._src.lib import xla_client as xc
from jax._src.lib import xla_extension as xe
from jax._src.sharding import Sharding
from jax._src.sharding_impls import (
    PmapSharding, SingleDeviceSharding, NamedSharding,
    device_replica_id_map, hashed_index, num_addressable_indices, local_to_global_shape)  # pyformat: disable
from jax._src.typing import ArrayLike, DLDeviceType
from jax._src.util import safe_zip, unzip3, use_cpp_class, use_cpp_method, cache
import numpy as np


Shape = tuple[int, ...]
Device = xc.Device
Index = tuple[slice, ...]
PRNGKeyArray = Any  # TODO(jakevdp): fix cycles and import this.

def _get_device(a: ArrayImpl) -> Device:
  devices = a.sharding._internal_device_list  # pytype: disable=attribute-error
  assert len(devices) == 1
  return devices[0]


class Shard:
  """A single data shard of an Array.

  Attributes:
    device : Which device this shard resides on.
    index : The index into the global array of this shard.
    replica_id : Integer id indicating which replica of the global array this
      shard is part of. Always 0 for fully sharded data
      (i.e. when there’s only 1 replica).
    data : The data of this shard. None if ``device`` is non-local.
  """

  def __init__(self, device: Device, sharding: Sharding, global_shape: Shape,
               data: None | ArrayImpl | PRNGKeyArray = None):
    self._device = device
    self._sharding = sharding
    self._global_shape = global_shape
    self._data = data

  def __repr__(self):
    try:
      return (f'Shard(device={self.device!r}, index={self.index}, '
              f'replica_id={self.replica_id}, data={self.data})')
    except ValueError:
      return f'Shard(device={self.device!r}, data={self.data})'

  @functools.cached_property
  def index(self) -> Index:
    try:
      device_indices_map_fn = self._sharding.devices_indices_map
    except AttributeError:
      raise ValueError('Cannot calculate indices from sharding: '
                       f'{self._sharding}. Please create a device to index '
                       'mapping for your sharding.') from None
    index = device_indices_map_fn(self._global_shape)[self.device]
    assert index is not None
    return index

  @functools.cached_property
  def replica_id(self) -> int:
    return device_replica_id_map(self._sharding, self._global_shape)[self.device]

  @property
  def device(self):
    return self._device

  @property
  def data(self):
    return self._data


def _reconstruct_array(fun, args, arr_state, aval_state):
  """Method to reconstruct a device array from a serialized state."""
  np_value = fun(*args)
  np_value.__setstate__(arr_state)
  jnp_value = api.device_put(np_value)
  # TODO(slebedev): Remove this branch after December 10th 2024.
  if "named_shape" in aval_state:
    deprecations.warn(
        "jax-aval-named-shape",
        "Pickled array contains an aval with a named_shape attribute. This is"
        " deprecated and the code path supporting such avals will be removed."
        " Please re-pickle the array.",
        stacklevel=2,
    )
    del aval_state["named_shape"]
  jnp_value.aval = jnp_value.aval.update(**aval_state)
  return jnp_value


@cache(max_size=4096, trace_context_in_key=False)
def _cached_index_calc(s, shape):
  map_ = s.addressable_devices_indices_map(shape)
  seen_h_indices = set()
  l = []
  for array_index, index in enumerate(map_.values()):
    h_index = hashed_index(index)
    if h_index not in seen_h_indices:
      seen_h_indices.add(h_index)
      l.append((array_index, index))
  return l


@cache(max_size=4096, trace_context_in_key=False)
def _process_has_full_value_in_mcjax(s, shape):
  # Return False for single host as a fast path.
  if xla_bridge.process_count() == 1:
    return False

  num_unique_indices = len(
      {hashed_index(v) for v in s.devices_indices_map(shape).values()})
  num_addressable_unique_indices = len(
      {hashed_index(v) for v in s.addressable_devices_indices_map(shape).values()})
  return num_unique_indices == num_addressable_unique_indices


def _validate_shape_and_dtype_for_per_device_arrays(
    arrays: Sequence[ArrayImpl | np.ndarray],
    sharding: Sharding,
    aval: core.ShapedArray,
    expected_shape: Shape,
):
  """Validates that per-device arrays are valid and consistent."""
  expected_dtype = aval.dtype
  for db in arrays:
    if db.dtype != expected_dtype:
      raise ValueError(
          "Input buffers to `Array` must have matching dtypes. "
          f"Got {db.dtype}, expected {expected_dtype} for buffer: {db}"
      )
    if db.shape != expected_shape:
      raise ValueError(
          f"Expected shard shape {expected_shape} doesn't match the single "
          f"device array shape {db.shape}. Shape of Array is "
          f"{aval.str_short()} with sharding {sharding}"
      )


class ArrayImpl(basearray.Array):
  # TODO(yashkatariya): Add __slots__ here.

  aval: core.ShapedArray
  _sharding: Sharding
  _arrays: list[ArrayImpl]
  _committed: bool
  _skip_checks: bool
  _npy_value: np.ndarray | None

  @use_cpp_method()
  def __init__(self, aval: core.ShapedArray, sharding: Sharding,
               arrays: Sequence[ArrayImpl],
               committed: bool, _skip_checks: bool = False):
    # NOTE: the actual implementation of the constructor is moved to C++.

    self.aval = aval
    self._sharding = sharding
    self._arrays = [a._arrays[0] for a in arrays]
    self._committed = committed
    self._npy_value = None

    # Don't rearrange if skip_checks is enabled because this assumes that the
    # input buffers are already arranged properly. This usually happens when
    # Array's are created as output of a JAX transformation
    # (like pjit, etc).
    if not _skip_checks or config.enable_checks.value:
      self._check_and_rearrange()

  def _check_and_rearrange(self):
    device_id_to_buffer = {_get_device(db).id: db for db in self._arrays}

    addressable_dev = self.sharding.addressable_devices
    if len(self._arrays) != len(addressable_dev):
      raise ValueError(
          f"Expected {len(addressable_dev)} per-device arrays "
          "(this is how many devices are addressable by the sharding), but "
          f"got {len(self._arrays)}")

    array_device_ids = set(device_id_to_buffer.keys())
    addressable_device_ids = {d.id for d in addressable_dev}
    # Calculate a symmetric difference because the device ids between sharding
    # and _arrays should match.
    diff = array_device_ids ^ addressable_device_ids
    if diff:
      dev_in_sharding_not_in_arrays = addressable_device_ids - array_device_ids
      dev_in_arrays_not_in_sharding = array_device_ids - addressable_device_ids
      err_msg = (
          "Addressable devices and per-device arrays devices do not match.")
      if dev_in_sharding_not_in_arrays:
        err_msg += (f" Sharding contains devices {dev_in_sharding_not_in_arrays} "
                    "that are not present in per-device arrays.")
      if dev_in_arrays_not_in_sharding:
        err_msg += (f" Per-device arrays contain devices {dev_in_arrays_not_in_sharding} "
                    "that are not present in the sharding.")
      raise ValueError(err_msg)

    _validate_shape_and_dtype_for_per_device_arrays(
        self._arrays,
        sharding=self.sharding,
        aval=self.aval,
        expected_shape=self.sharding.shard_shape(self.shape),
    )
    # Rearrange arrays based on the device assignment.
    addressable_da = self.sharding._addressable_device_assignment
    self._arrays = [device_id_to_buffer[device.id] for device in addressable_da]

  @property
  def shape(self) -> Shape:
    return self.aval.shape

  @property
  def dtype(self):
    return self.aval.dtype

  @property
  def ndim(self):
    return len(self.shape)

  @property
  def size(self):
    return math.prod(self.shape)

  @property
  def sharding(self):
    return self._sharding

  @property
  def device(self):
    self._check_if_deleted()
    if isinstance(self.sharding, SingleDeviceSharding):
      return list(self.sharding.device_set)[0]
    return self.sharding

  @property
  def weak_type(self):
    return self.aval.weak_type

  @property
  def committed(self) -> bool:
    return self._committed

  def __str__(self):
    return str(self._value)

  def __len__(self):
    try:
      return self.shape[0]
    except IndexError as err:
      raise TypeError("len() of unsized object") from err  # same as numpy error

  def __bool__(self):
    core.check_bool_conversion(self)
    return bool(self._value)

  def __float__(self):
    core.check_scalar_conversion(self)
    return self._value.__float__()

  def __int__(self):
    core.check_scalar_conversion(self)
    return self._value.__int__()

  def __complex__(self):
    core.check_scalar_conversion(self)
    return self._value.__complex__()

  def __hex__(self):
    core.check_integer_conversion(self)
    return hex(self._value)

  def __oct__(self):
    core.check_integer_conversion(self)
    return oct(self._value)

  def __index__(self):
    core.check_integer_conversion(self)
    return op.index(self._value)

  def tobytes(self, order="C"):
    return self._value.tobytes(order)

  def tolist(self):
    return self._value.tolist()

  def __format__(self, format_spec):
    # Simulates behavior of https://github.com/numpy/numpy/pull/9883
    if self.ndim == 0:
      return format(self._value[()], format_spec)
    else:
      return format(self._value, format_spec)

  def __getitem__(self, idx):
    from jax._src.lax import lax
    from jax._src.numpy import lax_numpy
    self._check_if_deleted()

    if isinstance(self.sharding, PmapSharding):
      if config.pmap_no_rank_reduction.value:
        cidx = idx if isinstance(idx, tuple) else (idx,)

        padded_cidx = tuple(
            slice(i, i + 1, None) if isinstance(i, int) else i for i in cidx
        ) + (slice(None),) * (len(self.shape) - len(cidx))
      else:
        if not isinstance(idx, tuple):
          padded_cidx = (idx,) + (slice(None),) * (len(self.shape) - 1)
        else:
          padded_cidx = idx + (slice(None),) * (len(self.shape) - len(idx))

      indices = tuple(self.sharding.devices_indices_map(self.shape).values())
      try:
        arr_idx = indices.index(padded_cidx)
      except ValueError:
        arr_idx = None
      if arr_idx is not None:
        out = self._arrays[arr_idx]
        sharding = SingleDeviceSharding(_get_device(out))

        if config.pmap_no_rank_reduction.value:
          # If cidx was the index of a single shard, then it corresponds to one
          # shard of the chunked dimension.
          dims = tuple(i for i, x in enumerate(cidx) if isinstance(x, int))
          # Squeeze on committed arrays to avoid data movement to shard 0.
          out = lax.squeeze(out, dimensions=dims)

        return ArrayImpl(
            out.aval, sharding, [out], committed=False, _skip_checks=True)

    return lax_numpy._rewriting_take(self, idx)

  def __iter__(self):
    if self.ndim == 0:
      raise TypeError("iteration over a 0-d array")  # same as numpy error
    else:
      assert self.is_fully_replicated or self.is_fully_addressable
      if dispatch.is_single_device_sharding(self.sharding) or self.is_fully_replicated:
        return (sl for chunk in self._chunk_iter(100) for sl in chunk._unstack())
      elif isinstance(self.sharding, PmapSharding):
        return (self[i] for i in range(self.shape[0]))
      else:
        # TODO(yashkatariya): Don't bounce to host and use `_chunk_iter` path
        # here after uneven partitioning support is added.
        return (api.device_put(self._value[i]) for i in range(self.shape[0]))

  @property
  def is_fully_replicated(self) -> bool:
    return self.sharding.is_fully_replicated

  def __repr__(self):
    prefix = 'Array('
    if self.aval is not None and self.aval.weak_type:
      dtype_str = f'dtype={self.dtype.name}, weak_type=True)'
    else:
      dtype_str = f'dtype={self.dtype.name})'

    if self.is_fully_addressable or self.is_fully_replicated:
      line_width = np.get_printoptions()["linewidth"]
      if self.size == 0:
        s = f"[], shape={self.shape}"
      else:
        s = np.array2string(self._value, prefix=prefix, suffix=',',
                            separator=', ', max_line_width=line_width)
      last_line_len = len(s) - s.rfind('\n') + 1
      sep = ' '
      if last_line_len + len(dtype_str) + 1 > line_width:
        sep = ' ' * len(prefix)
      return f"{prefix}{s},{sep}{dtype_str}"
    else:
      return f"{prefix}{self.shape}, {dtype_str}"

  @property
  def is_fully_addressable(self) -> bool:
    """Is this Array fully addressable?

    A jax.Array is fully addressable if the current process can address all of
    the devices named in the :class:`Sharding`. ``is_fully_addressable`` is
    equivalent to "is_local" in multi-process JAX.

    Note that fully replicated is not equal to fully addressable i.e.
    a jax.Array which is fully replicated can span across multiple hosts and is
    not fully addressable.
    """
    return self.sharding.is_fully_addressable

  def __array__(self, dtype=None, context=None, copy=None):
    # copy argument is supported by np.asarray starting in numpy 2.0
    kwds = {} if copy is None else {'copy': copy}
    return np.asarray(self._value, dtype=dtype, **kwds)

  def __dlpack__(self, *, stream: int | Any | None = None,
                 max_version: tuple[int, int] | None = None,
                 dl_device: tuple[DLDeviceType, int] | None = None,
                 copy: bool | None = None):
    from jax._src.dlpack import to_dlpack  # pylint: disable=g-import-not-at-top

    device_set = self.sharding.device_set
    if len(device_set) > 1:
      raise BufferError(
        "to_dlpack can only pack a dlpack tensor from an array on a singular "
        f"device, but an array with a Sharding over {len(device_set)} devices "
        "was provided."
      )
    device, = device_set
    return to_dlpack(self, stream=stream,
                     max_version=max_version,
                     src_device=device,
                     dl_device=dl_device,
                     copy=copy)

  def __dlpack_device__(self) -> tuple[enum.Enum, int]:
    if len(self._arrays) != 1:
      raise BufferError("__dlpack__ only supported for unsharded arrays.")

    from jax._src.dlpack import DLDeviceType  # pylint: disable=g-import-not-at-top

    if self.platform() == "cpu":
      return DLDeviceType.kDLCPU, 0

    elif self.platform() == "gpu":
      platform_version = _get_device(self).client.platform_version
      if "cuda" in platform_version:
        dl_device_type = DLDeviceType.kDLCUDA
      elif "rocm" in platform_version:
        dl_device_type = DLDeviceType.kDLROCM
      else:
        raise BufferError("Unknown GPU platform for __dlpack__: "
                         f"{platform_version}")

      local_hardware_id = _get_device(self).local_hardware_id
      if local_hardware_id is None:
        raise BufferError("Couldn't get local_hardware_id for __dlpack__")

      return dl_device_type, local_hardware_id

    else:
      raise BufferError(
          "__dlpack__ device only supported for CPU and GPU, got platform: "
          f"{self.platform()}"
      )

  def __reduce__(self):
    fun, args, arr_state = self._value.__reduce__()
    aval_state = {'weak_type': self.aval.weak_type}
    return (_reconstruct_array, (fun, args, arr_state, aval_state))

  @use_cpp_method()
  def unsafe_buffer_pointer(self):
    if len(self._arrays) != 1:
      raise ValueError("unsafe_buffer_pointer() is supported only for unsharded"
                       " arrays.")
    return self._arrays[0].unsafe_buffer_pointer()

  @property
  @use_cpp_method()
  def __cuda_array_interface__(self):
    if len(self._arrays) != 1:
      raise ValueError("__cuda_array_interface__() is supported only for "
                       "unsharded arrays.")
    return self._arrays[0].__cuda_array_interface__  # pytype: disable=attribute-error  # bind-properties

  @use_cpp_method()
  def on_device_size_in_bytes(self):
    """Returns the total global on-device size of the array in bytes."""
    arr = self._arrays[0]
    per_shard_size = arr.on_device_size_in_bytes()
    return per_shard_size * self.sharding.num_devices

  def devices(self) -> set[Device]:
    self._check_if_deleted()
    return self.sharding.device_set

  @property
  def device_buffer(self):
    raise AttributeError(
      "arr.device_buffer has been deprecated. Use arr.addressable_data(0)")

  @property
  def device_buffers(self):
    raise AttributeError(
      "arr.device_buffers has been deprecated. Use [x.data for x in arr.addressable_shards]")

  def addressable_data(self, index: int) -> ArrayImpl:
    self._check_if_deleted()
    if self.is_fully_replicated:
      return self._fully_replicated_shard()
    return self._arrays[index]

  @functools.cached_property
  def addressable_shards(self) -> Sequence[Shard]:
    self._check_if_deleted()
    out = []
    for a in self._arrays:
      out.append(Shard(_get_device(a), self.sharding, self.shape, a))
    return out

  @property
  def layout(self):
    # TODO(yashkatariya): Remove the deleted check from here.
    if self.is_deleted():
      return Layout(None, self.sharding)
    try:
      return Layout(DeviceLocalLayout.from_pjrt_layout(self._pjrt_layout),
                    self.sharding)
    except xe.XlaRuntimeError as e:
      msg, *_ = e.args
      if type(msg) is str and msg.startswith("UNIMPLEMENTED"):
        return Layout(None, self.sharding)
      else:
        raise

  @property
  def global_shards(self) -> Sequence[Shard]:
    """Returns list of all `Shard`s of the Array across all devices.

    The result includes shards that are not addressable by the current process.
    If a `Shard` is not addressable, then its `data` will be `None`.
    """
    self._check_if_deleted()
    if self.is_fully_addressable:  # pylint: disable=using-constant-test
      return self.addressable_shards

    out = []
    device_id_to_buffer = {_get_device(a).id: a for a in self._arrays}
    for global_d in self.sharding.device_set:
      if device_id_to_buffer.get(global_d.id, None) is not None:
        array = device_id_to_buffer[global_d.id]
      else:
        array = None
      out.append(Shard(global_d, self.sharding, self.shape, array))
    return out

  @use_cpp_method()
  def delete(self):
    if self._arrays is None:
      return
    for buf in self._arrays:
      buf.delete()
    self._arrays = None
    self._npy_value = None

  @use_cpp_method()
  def is_deleted(self):
    if self._arrays is None:
      return True
    # This path is taken when a view of `Array` is created and the original
    # Array is deleted. In that case, the buffers the view represents also get
    # deleted.
    return any(buf.is_deleted() for buf in self._arrays)

  def _check_if_deleted(self):
    if self.is_deleted():
      raise RuntimeError(
          f"Array has been deleted with shape={self.aval.str_short()}.")

  @use_cpp_method()
  def block_until_ready(self):
    self._check_if_deleted()
    for db in self._arrays:
      db.block_until_ready()
    return self

  @use_cpp_method()
  def _single_device_array_to_np_array(self):
    return np.asarray(self._arrays[0])

  @use_cpp_method()
  def _copy_single_device_array_to_host_async(self):
    self._arrays[0].copy_to_host_async()

  @profiler.annotate_function
  def copy_to_host_async(self):
    self._check_if_deleted()
    if self._npy_value is None:
      if self.is_fully_replicated:
        self._copy_single_device_array_to_host_async()
        return
      for i, _ in _cached_index_calc(self.sharding, self.shape):
        self._arrays[i]._copy_single_device_array_to_host_async()

  @property
  @functools.partial(profiler.annotate_function, name="np.asarray(jax.Array)")
  def _value(self) -> np.ndarray:
    self._check_if_deleted()

    if self._npy_value is None:
      if self.is_fully_replicated:
        self._npy_value = self._single_device_array_to_np_array()
        self._npy_value.flags.writeable = False
        return cast(np.ndarray, self._npy_value)

      # TODO(yashkatariya): Merge `_process_has_full_value_in_mcjax` with
      # is_fully_addressable.
      if (not self.is_fully_addressable and
          not _process_has_full_value_in_mcjax(self.sharding, self.shape)):
        raise RuntimeError(
            "Fetching value for `jax.Array` that spans non-addressable"
            " (non process local) devices is not possible. You can use"
            " `jax.experimental.multihost_utils.process_allgather` to print the"
            " global array or use `.addressable_shards` method of jax.Array to"
            " inspect the addressable (process local) shards."
        )

      for i, _ in _cached_index_calc(self.sharding, self.shape):
        self._arrays[i]._copy_single_device_array_to_host_async()

      npy_value = np.empty(self.shape, self.dtype)
      for i, ind in _cached_index_calc(self.sharding, self.shape):
        npy_value[ind] = self._arrays[i]._single_device_array_to_np_array()
      self._npy_value = npy_value
      self._npy_value.flags.writeable = False
    return self._npy_value


# TODO(b/273265390): ideally we would write this as a decorator on the ArrayImpl
# class, however this triggers a pytype bug. Workaround: apply the decorator
# after the fact.
if not TYPE_CHECKING:
  ArrayImpl = use_cpp_class(xc.ArrayImpl)(ArrayImpl)


def _get_shape_from_index(slc: Index, shape: Shape) -> Shape:
  return tuple(
      (s.stop or dim) - (s.start or 0)
      for s, dim in safe_zip(slc, shape)
      if isinstance(s, slice)  # If element is int, this dimension is reduced
  )


# explicitly set to be unhashable.
setattr(ArrayImpl, "__hash__", None)
setattr(ArrayImpl, "__array_priority__", 100)

# TODO(yashkatariya): Remove None from callback input type.

def make_array_from_callback(
    shape: Shape, sharding: Sharding | Layout,
    data_callback: Callable[[Index | None], ArrayLike]) -> ArrayImpl:
  # pyformat: disable
  """Returns a ``jax.Array`` via data fetched from ``data_callback``.

  ``data_callback`` is used to fetch the data for each addressable shard of the
  returned ``jax.Array``. This function must return concrete arrays, meaning that
  ``make_array_from_callback`` has limited compatibility with JAX transformations
  like :func:`jit` or :func:`vmap`.

  Args:
    shape : Shape of the ``jax.Array``.
    sharding: A ``Sharding`` instance which describes how the ``jax.Array`` is
      laid out across devices.
    data_callback : Callback that takes indices into the global array value as
      input and returns the corresponding data of the global array value.
      The data can be returned as any array-like object, e.g. a ``numpy.ndarray``.

  Returns:
    A ``jax.Array`` via data fetched from ``data_callback``.

  Examples:

    >>> import math
    >>> from jax.sharding import Mesh
    >>> from jax.sharding import PartitionSpec as P
    >>> import numpy as np
    ...
    >>> input_shape = (8, 8)
    >>> global_input_data = np.arange(math.prod(input_shape)).reshape(input_shape)
    >>> global_mesh = Mesh(np.array(jax.devices()).reshape(2, 4), ('x', 'y'))
    >>> inp_sharding = jax.sharding.NamedSharding(global_mesh, P('x', 'y'))
    ...
    >>> def cb(index):
    ...  return global_input_data[index]
    ...
    >>> arr = jax.make_array_from_callback(input_shape, inp_sharding, cb)
    >>> arr.addressable_data(0).shape
    (4, 2)
  """
  # pyformat: enable
  dll = sharding.device_local_layout if isinstance(sharding, Layout) else None
  if isinstance(dll, AutoLayout):
    raise TypeError(
        "`DeviceLocalLayout.AUTO` cannot be used in place of a device-local"
        f" layout when calling `jax.make_array_from_callback`. Got {sharding}")
  sharding = sharding.sharding if isinstance(sharding, Layout) else sharding  # type: ignore
  if not isinstance(sharding, Sharding):
    raise TypeError(
        f"sharding should be an instance of `jax.sharding`. Got {sharding} of"
        f" type {type(sharding)}")

  def get_data(index: Index | None) -> ArrayImpl | np.ndarray:
    # Perhaps cache on index here, then we can unify fully_replicated
    # and non-fully_replicated cases below and become faster for
    # partially replicated cases.
    assert index is not None
    r = data_callback(index)
    if isinstance(r, core.Tracer):
      raise errors.UnexpectedTracerError(
          "jax.make_array_from_callback cannot be called within a traced"
          " context."
      )
    # Value can be python scalar, resolve it into something with dtype.
    return xla.canonicalize_dtype(r)

  if sharding.is_fully_replicated:
    devices = list(sharding._internal_device_list.addressable_device_list)  # type: ignore
    # Only compute data once.
    per_device_values = [get_data((slice(None),) * len(shape))] * len(devices)
  else:
    device_to_index_map = sharding.addressable_devices_indices_map(shape)
    devices = list(device_to_index_map.keys())
    per_device_values = [
        get_data(device_to_index_map[device]) for device in devices
    ]

  first_value = per_device_values[0]
  expected_dtype = first_value.dtype
  expected_shape = sharding.shard_shape(shape)
  aval = core.ShapedArray(shape, expected_dtype)
  _validate_shape_and_dtype_for_per_device_arrays(
      per_device_values,
      expected_shape=expected_shape,
      aval=aval,
      sharding=sharding,
  )
  if (isinstance(first_value, ArrayImpl)
      and first_value._committed
      and sharding.is_fully_replicated
      and first_value.is_fully_replicated
      and first_value.sharding._device_assignment == tuple(devices)
      and first_value.layout.device_local_layout == dll):
    return first_value

  if dtypes.issubdtype(aval.dtype, dtypes.extended):
    # TODO(yashkatariya): Can this also use batched_device_put?
    arrays = api.device_put(per_device_values, devices)
    return aval.dtype._rules.make_sharded_array(
        aval, sharding, arrays, committed=True
    )

  if dll is not None:
    devices = [Layout(dll, SingleDeviceSharding(d)) for d in devices]
    # pxla.batched_device_put doesn't support Layout... Take the slow route
    arrays = api.device_put(per_device_values, devices)
    return ArrayImpl(aval, sharding, arrays, committed=True)

  if isinstance(first_value, ArrayImpl) and len(first_value.devices()) > 1:
    # The output of the callback is already a sharded array, move it to
    # to target device.
    per_device_values = api.device_put(per_device_values, devices)

  return pxla.batched_device_put(aval, sharding, per_device_values, devices)


def make_array_from_process_local_data(
    sharding: Sharding,
    local_data: np.ndarray,
    global_shape: Shape | None = None,
) -> ArrayImpl:
  # pyformat: disable
  """Creates distributed tensor using the data available in process.

  This function is a common special case of `make_array_from_callback`. It
  assumes that the data is available in the process and takes care of the
  index wrangling.

  The most common case is when the sharding is sharded across the batch
  dimension and each host just loads its corresponding sub-batch. This function
  supports more general cases as well, such as mixed multi-host and multi-axis
  replication and sharding but you would need to compute the size and the
  contents of process-local data correctly to satisfy the sharding constraints.

  In particular, if any two hosts are replicas, host_local_data should be
  identical as well.

  The global_shape is optional. If not provided it will be be inferred from
  the local_data and sharding, under the assumption that
  each host represents only their own data for uniform sharding. If sharding
  is non-uniform, (see note below) an exception will be raised.

  Setting global_shape explicitly allows for finer grain control and works with
  non-uniform shardings. Each dimension of global_shape must either match
  host_local_data, or match the inferred global shape of the sharding (in which
  case it is equivalent to setting it to None, but is more explicit).

  For example if dimension `i` is fully sharded then this size would be
  `per_device_shape[i] * jax.local_device_count()`.  Each device will be mapped
  into local slice of `local_data` array. For example, if given process
  addresses slices (8, 12) and  (24, 28), then these slices will be mapped
  into (0, 4) and (4, 8) of the `local_data`.

  For each dimension where global_shapes matches local_shape, each device
  will lookup the slice in the local_data. For example if
  global_shape == local_data.shape, the local data is assumed to be the
  actual target array that will be sharded into device.

  If global_shape is the same as local_data.shape, then the data must
  be the same across all hosts.

  Examples:
    >>> from jax.sharding import PartitionSpec as P
    >>> mesh_rows = 2
    >>> mesh_cols =  jax.device_count() // 2
    ...
    >>> mesh = jax.sharding.Mesh(np.array(jax.devices()).reshape(mesh_rows, mesh_cols), ('x', 'y'))

    >>> sharding = jax.sharding.NamedSharding(mesh, P(('x', 'y'),))
    >>> rows_per_device = 2
    >>> feature_length = 32
    >>> per_device_shape = (rows_per_device, feature_length)
    >>> per_host_shape = (rows_per_device * len(mesh.local_devices), feature_length)
    >>> per_host_generator = lambda : np.arange(np.prod(per_host_shape)).reshape(per_host_shape)
    >>> per_host_data = per_host_generator()  # replace with your own per-host data pipeline that outputs numpy arrays
    >>> global_shape = (rows_per_device * len(sharding.device_set), ) + per_device_shape[1:]
    >>> output_global_array = jax.make_array_from_process_local_data(sharding, per_host_data, global_shape)
    ...
    >>> assert output_global_array.addressable_data(0).shape == per_device_shape
    >>> assert output_global_array.shape == global_shape

  NB: While most shardings are uniform, It is possible to design am exotic
  sharding mesh where each process's  devices will be arranged in a non-grid
  like pattern in some dimensions, or for indices to overlap non-trivially.
  Such sharding is called "non-uniform" in those dimensions. In that case,
  the global shape along those directions must match local shape as there is
  no meaningful way to represent all needed
  per-process data in non-overlapping fashion. For example for global_shape 4x4
  if sharding looks like this:

      0123
      2103
      4675
      4567

  with 4 processes, containing devices (0,1), (2, 3), (4, 5), (6, 7) respectively.
  Then the data for each host look like

      xx..    ..xx     ....    ....
      .xx.    x..x     ....    ....
      ....    ....     x..x    .xx.
      ....    ....     xx..    ..xx

  the sharding is uniform on rows (each host requires either rows 1-2, or rows 3-4)
  and non-uniform on columns (hosts require overlapping but not matching
  set of columns). Thus local data must have the shape 2x4 or 4x4
  for all hosts, even though each  host can potentially fit into 2x2 shape.
  In this case user must provide global_shape explicitly and for
  local_shape=(2, 4), potentially valid global shapes are (2, 4) and (4, 4).

  On the other hand for sharding:

      0213   x.x.  .x.x.  ....  ....
      0213   x.x.  .x.x.  ....  ....
      4657   ....  ....   .x.x  x.x.
      4657   ....  ....   .x.x  x.x.

  for local_shape=(2, 2) this function can accept a choice of 2x2, 2x4, 4x2
  and 4x4 global shapes. Setting global_shape to None, is equivalent to
  setting it to (4, 4) in this case.

  Args:
    sharding: Sharding of the global array.
    local_data: Data on the host to be placed on local devices. Each
      dimension should either match global_shape, or match
      num_addressable_indices(dim).
    global_shape: The target shape of the global array. If None,
      will infer from local_data and sharding.

  Returns:
    Tensor that will have sharding=sharding and of shape global_shape.
  """
  # pyformat: enable
  if xla_bridge.process_count() == 1:
    return api.device_put(local_data, sharding)

  # TODO(sandler): consider supporting partially specified global_shape or
  # making local_to_global_shape available in the api.
  local_shape = local_data.shape
  if global_shape is None:
    global_shape = local_to_global_shape(sharding, local_shape)  # type: ignore[assignment]
    assert global_shape is not None
    if None in global_shape:
      raise ValueError(
          "Unable to compute global_shape due to non-uniform sharding."
          f" Specify global shape directly. Partially computed {global_shape=}."
      )
  elif None in global_shape:
    raise ValueError(f"{global_shape=} has Nones. This is not supported.")
  full_dim = []
  for i, (data_dim, global_dim) in enumerate(
      zip(local_data.shape, global_shape)
  ):
    full_dim.append(data_dim == global_dim)
    if data_dim != global_dim:
      process_slice = num_addressable_indices(sharding, i, global_shape)
      if process_slice != data_dim:
        raise ValueError(
            "Invalid host data, each dimension should match either global or "
            f"process shape. In dimension {i=}, the process data has {data_dim}"
            f"elements. Process addresses {process_slice} elements and "
            f"{global_shape=}."
        )
  addressable_shards = sharding.addressable_devices_indices_map(global_shape)
  shard = next(iter(addressable_shards.values()))
  assert shard is not None
  shard_shape = _get_shape_from_index(shard, global_shape)
  slices_for_each_dim: list[list[int]] = [[] for _ in global_shape]
  for shard_index in addressable_shards.values():
    assert shard_index is not None
    for i, slc in enumerate(shard_index):
      slices_for_each_dim[i].append(slc.start or 0)
  for i in range(len(global_shape)):
    slices_for_each_dim[i] = sorted(set(slices_for_each_dim[i]))

  @functools.lru_cache(maxsize=4096)
  def local_slice(i, start):
    # Looks up the index of this slice in the list of slices for this dimension.
    # This will determine the slice in host_local_data
    start = slices_for_each_dim[i].index(start or 0) * shard_shape[i]
    end = start + shard_shape[i]
    return slice(start, end)

  def cb(index: Index | None) -> ArrayLike:
    assert index is not None
    data_slice = (
        slc if full_dim[i] else local_slice(i, slc.start)
        for i, slc in enumerate(index)
    )
    return local_data[tuple(data_slice)]

  return make_array_from_callback(global_shape, sharding, cb)


def make_array_from_single_device_arrays(
    shape: Shape, sharding: Sharding, arrays: Sequence[basearray.Array]
) -> ArrayImpl:
  r"""Returns a ``jax.Array`` from a sequence of ``jax.Array``\s each on a single device.
      Every device in input ``sharding``\'s mesh must have an array in ``arrays``\s.

  Args:
    shape : Shape of the output ``jax.Array``. This conveys information already included with
      ``sharding`` and ``arrays`` and serves as a double check.
    sharding: Sharding: A global Sharding instance which describes how the output jax.Array is laid out across devices.
    arrays: Sequence of ``jax.Array``\s that are each single device addressable. ``len(arrays)``
      must equal ``len(sharding.addressable_devices)`` and the shape of each array must be the same. For multiprocess code,
      each process will call with a different ``arrays`` argument that corresponds to that processes' data.
      These arrays are commonly created via ``jax.device_put``.

  Returns:
    A global ``jax.Array``, sharded as ``sharding``, with shape equal to ``shape``, and with per-device
      contents matching ``arrays``.

  Examples:

    >>> import math
    >>> from jax.sharding import Mesh
    >>> from jax.sharding import PartitionSpec as P
    >>> import numpy as np
    ...
    >>> mesh_rows = 2
    >>> mesh_cols =  jax.device_count() // 2
    ...
    >>> global_shape = (8, 8)
    >>> mesh = Mesh(np.array(jax.devices()).reshape(mesh_rows, mesh_cols), ('x', 'y'))
    >>> sharding = jax.sharding.NamedSharding(mesh, P('x', 'y'))
    >>> inp_data = np.arange(math.prod(global_shape)).reshape(global_shape)
    ...
    >>> arrays = [
    ...    jax.device_put(inp_data[index], d)
    ...        for d, index in sharding.addressable_devices_indices_map(global_shape).items()]
    ...
    >>> arr = jax.make_array_from_single_device_arrays(global_shape, sharding, arrays)
    >>> assert arr.shape == (8,8) # arr.shape is (8,8) regardless of jax.device_count()

  For cases where you have a local array and want to convert it to a global
  jax.Array, use ``jax.make_array_from_process_local_data``.
  """
  # All input arrays should be committed. Checking it is expensive on
  # single-controller systems.
  if any(isinstance(arr, core.Tracer) for arr in arrays):
    raise ValueError(
        "jax.make_array_from_single_device_arrays requires a list of concrete"
        f" arrays as input. got types {set(map(type, arrays))}")
  aval = core.ShapedArray(shape, arrays[0].dtype, weak_type=False)
  if dtypes.issubdtype(aval.dtype, dtypes.extended):
    return aval.dtype._rules.make_sharded_array(aval, sharding, arrays,
                                                committed=True)
  # TODO(phawkins): ideally the cast() could be checked.
  return ArrayImpl(aval, sharding, cast(Sequence[ArrayImpl], arrays),
                   committed=True)

xla.canonicalize_dtype_handlers[ArrayImpl] = pxla.identity

def _get_aval_array(self):
  if config.sharding_in_types.value and isinstance(self.sharding, NamedSharding):
    return self.aval.update(sharding=NamedSharding(
        self.sharding.mesh.abstract_mesh,
        self.sharding.spec._normalized_spec(self.ndim)))
  else:
    return self.aval

core.pytype_aval_mappings[ArrayImpl] = _get_aval_array

# TODO(jakevdp) replace this with true inheritance at the C++ level.
basearray.Array.register(ArrayImpl)


def _array_mlir_constant_handler(val):
  try:
    return mlir.ir_constant(val._value)
  except RuntimeError as e:
    # TODO(yashkatariya): Ideally we would catch a custom exception from
    # `_value` function in ArrayImpl instead of checking the error string.
    if 'Fetching value for `jax.Array` that spans non-addressable' in str(e):
      raise RuntimeError(
          "Closing over jax.Array that spans non-addressable (non process"
          " local) devices is not allowed. Please pass such arrays as arguments"
          f" to the function. Got jax.Array: {val.aval.str_short()}") from e
    raise

mlir.register_constant_handler(ArrayImpl, _array_mlir_constant_handler)


# NOTE(skye): we could refactor to generate _multi_slice parameters directly
# from the input ShardingSpec, rather than the indices. However, this would
# require duplicating the ordering logic of spec_to_indices, which is more
# subtle and more likely to change than the index logic we have to support here.
def as_slice_indices(arr: Any, idx: Index) -> tuple[
    tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
  """Returns start_indices, limit_indices, removed_dims"""
  start_indices = [0] * arr.ndim
  limit_indices = list(arr.shape)
  removed_dims: list[int] = []

  tuple_idx = idx if isinstance(idx, tuple) else (idx,)
  for dim, sub_idx in enumerate(tuple_idx):
    if isinstance(sub_idx, int):
      start_indices[dim] = sub_idx
      limit_indices[dim] = sub_idx + 1
      removed_dims.append(dim)
    elif sub_idx == slice(None):
      continue
    else:
      assert isinstance(sub_idx, slice), sub_idx
      assert isinstance(sub_idx.start, int), sub_idx
      assert isinstance(sub_idx.stop, int), sub_idx
      start_indices[dim] = sub_idx.start
      limit_indices[dim] = sub_idx.stop

  return tuple(start_indices), tuple(limit_indices), tuple(removed_dims)


def shard_device_array(x, devices, indices, sharding):
  start_indices, limit_indices, removed_dims = unzip3(
      as_slice_indices(x, idx) for idx in indices)
  if sharding.is_fully_replicated:
    shards = [x] * len(devices)
  else:
    shards = x._multi_slice(start_indices, limit_indices, removed_dims)
  aval = core.shaped_abstractify(x)
  return pxla.batched_device_put(aval, sharding, shards, devices)


def shard_sharded_device_array_slow_path(x, devices, indices, sharding):
  candidates = defaultdict(list)
  bufs = [buf.data for buf in x.addressable_shards]
  arr_indices = tuple(x.sharding.devices_indices_map(x.shape).values())
  for buf, idx in safe_zip(bufs, arr_indices):
    candidates[hashed_index(idx)].append(buf)

  bufs = []
  for idx, device in safe_zip(indices, devices):
    # Look up all buffers that contain the correct slice of the logical array.
    candidates_list = candidates[hashed_index(idx)]
    if not candidates_list:
      return pxla.shard_args([sharding], [None], [None], [x._value],
                             canonicalize=False)[0]
    # Try to find a candidate buffer already on the correct device,
    # otherwise copy one of them.
    for buf in candidates_list:
      if buf.devices() == {device}:
        bufs.append(buf)
        break
    else:
      bufs.append(candidates_list[-1])
  return pxla.batched_device_put(x.aval, sharding, bufs, devices)


@cache(max_size=4096, trace_context_in_key=False)
def _sharding_indices_and_eq(src_sharding, shape, dst_sharding):
  src_indices = src_sharding.addressable_devices_indices_map(shape).values()
  dst_indices = dst_sharding.addressable_devices_indices_map(shape).values()
  return dst_indices, tuple(src_indices) == tuple(dst_indices)


def _array_shard_arg(xs, shardings, layouts, copy_semantics):
  util.test_event("_array_shard_arg")
  results = []
  batch_xs, batch_devs, batch_shardings, batch_indices = [], [], [], []
  batch_cs = []

  for i, (x, sharding, layout, cs) in enumerate(
      safe_zip(xs, shardings, layouts, copy_semantics)):
    x._check_if_deleted()
    indices, same_indices = _sharding_indices_and_eq(x.sharding, x.shape, sharding)
    same_layout = (True if layout is None else
                   x.layout.device_local_layout == layout)

    if not x.is_fully_addressable:
      if same_indices and same_layout:
        results.append(x)
      else:
        raise NotImplementedError(
            "Cannot reshard an input that is not fully addressable")
    else:
      devices = sharding._addressable_device_assignment
      if same_indices and same_layout:
        # Add a placeholder result that will be filled in later.
        results.append(None)
        # Accumulate arguments to `batched_copy_array_to_devices_with_sharding`.
        batch_xs.append(x)
        batch_devs.append(list(devices))
        batch_shardings.append(sharding)
        batch_indices.append(i)
        batch_cs.append(cs)
      # Resharding starts here:
      elif not same_layout:
        results.append(api.device_put(x, Layout(layout, sharding)))
      elif dispatch.is_single_device_sharding(x.sharding):
        results.append(shard_device_array(x, devices, indices, sharding))
      else:
        results.append(
            shard_sharded_device_array_slow_path(x, devices, indices, sharding))

  util.test_event("batched_copy_array")
  copy_outs = xc.batched_copy_array_to_devices_with_sharding(
      batch_xs, batch_devs, batch_shardings, batch_cs)
  for i, copy_out in safe_zip(batch_indices, copy_outs):
    assert results[i] is None
    results[i] = copy_out
  return results
pxla.shard_arg_handlers[ArrayImpl] = _array_shard_arg


def _array_global_result_handler(global_aval, out_sharding, committed):
  if global_aval.dtype == dtypes.float0:
    return lambda _: np.zeros(global_aval.shape, dtypes.float0)
  if dtypes.issubdtype(global_aval.dtype, dtypes.extended):
    return global_aval.dtype._rules.global_sharded_result_handler(
        global_aval, out_sharding, committed)
  return xc.array_result_handler(
      global_aval, out_sharding, committed=committed, _skip_checks=True
  )
pxla.global_result_handlers[core.ShapedArray] = _array_global_result_handler

# Only used for Arrays that come out of pmap.
def _array_local_result_handler(aval, sharding, indices):
  if aval.dtype == dtypes.float0:
    return lambda _: np.zeros(aval.shape, dtypes.float0)
  if dtypes.issubdtype(aval.dtype, dtypes.extended):
    return aval.dtype._rules.local_sharded_result_handler(
        aval, sharding, indices)
  return xc.array_result_handler(
      aval, sharding, committed=True, _skip_checks=True
  )
pxla.local_result_handlers[core.ShapedArray] = _array_local_result_handler


# Token handlers

def _token_shard_arg(xs, shardings, layouts, copy_semantics):
  results = []
  for x, sharding, layout in safe_zip(xs, shardings, layouts):
    x.block_until_ready()
    x = np.array([], dtype=bool)
    results.append(api.device_put(x, Layout(layout, sharding)))
  return results
pxla.shard_arg_handlers[core.Token] = _token_shard_arg


def _token_global_result_handler(global_aval, out_sharding, committed):
  array_handler = _array_global_result_handler(
      core.token_shaped_array, out_sharding, committed)

  def wrapper(*args, **kwargs):
    out_buf = array_handler(*args, **kwargs)
    return core.Token(out_buf)
  return wrapper
pxla.global_result_handlers[core.AbstractToken] = _token_global_result_handler
