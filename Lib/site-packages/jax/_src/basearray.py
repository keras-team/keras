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

# Note that type annotations for this file are defined in basearray.pyi

from __future__ import annotations

import abc
import numpy as np
from typing import Any, Union
from collections.abc import Sequence

# TODO(jakevdp): fix import cycles and define these.
Device = Any
Shard = Any
Sharding = Any

# Array is a type annotation for standard JAX arrays and tracers produced by
# core functions in jax.lax and jax.numpy; it is not meant to include
# future non-standard array types like KeyArray and BInt.
class Array(abc.ABC):
  """Array base class for JAX

  ``jax.Array`` is the public interface for instance checks and type annotation
  of JAX arrays and tracers. Its main applications are in instance checks and
  type annotations; for example::

    x = jnp.arange(5)
    isinstance(x, jax.Array)  # returns True both inside and outside traced functions.

    def f(x: Array) -> Array:  # type annotations are valid for traced and non-traced types.
      return x

  ``jax.Array`` should not be used directly for creation of arrays; instead you
  should use array creation routines offered in :mod:`jax.numpy`, such as
  :func:`jax.numpy.array`, :func:`jax.numpy.zeros`, :func:`jax.numpy.ones`,
  :func:`jax.numpy.full`, :func:`jax.numpy.arange`, etc.
  """
  # Note: abstract methods for this class are defined dynamically in
  # lax_numpy.py
  # For the sake of static type analysis, these definitions are mirrored in the
  # associated basearray.pyi file.

  __slots__ = ['__weakref__']
  __hash__ = None

  @property
  @abc.abstractmethod
  def dtype(self) -> np.dtype:
    """The data type (:class:`numpy.dtype`) of the array."""

  @property
  @abc.abstractmethod
  def ndim(self) -> int:
    """The number of dimensions in the array."""

  @property
  @abc.abstractmethod
  def size(self) -> int:
    """The total number of elements in the array."""

  @property
  @abc.abstractmethod
  def shape(self) -> tuple[int, ...]:
    """The shape of the array."""

  # Documentation for sharding-related methods and properties defined on ArrayImpl:
  @abc.abstractmethod
  def addressable_data(self, index: int) -> Array:
    """Return an array of the addressable data at a particular index."""

  @property
  @abc.abstractmethod
  def addressable_shards(self) -> Sequence[Shard]:
    """List of addressable shards."""

  @property
  @abc.abstractmethod
  def global_shards(self) -> Sequence[Shard]:
    """List of global shards."""

  @property
  @abc.abstractmethod
  def is_fully_addressable(self) -> bool:
    """Is this Array fully addressable?

    A jax.Array is fully addressable if the current process can address all of
    the devices named in the :class:`Sharding`. ``is_fully_addressable`` is
    equivalent to "is_local" in multi-process JAX.

    Note that fully replicated is not equal to fully addressable i.e.
    a jax.Array which is fully replicated can span across multiple hosts and is
    not fully addressable.
    """

  @property
  @abc.abstractmethod
  def is_fully_replicated(self) -> bool:
    """Is this Array fully replicated?"""

  @property
  @abc.abstractmethod
  def sharding(self) -> Sharding:
    """The sharding for the array."""

  @property
  @abc.abstractmethod
  def committed(self) -> bool:
    """Whether the array is committed or not.

    An array is committed when it is explicitly placed on device(s) via JAX
    APIs. For example, `jax.device_put(np.arange(8), jax.devices()[0])` is
    committed to device 0. While `jax.device_put(np.arange(8))` is uncommitted
    and will be placed on the default device.

    Computations involving some committed inputs will happen on the committed
    device(s) and the result will be committed on the same device(s).
    Invoking an operation on arguments that are committed to different device(s)
    will raise an error.

    For example:

    ```
    a = jax.device_put(np.arange(8), jax.devices()[0])
    b = jax.device_put(np.arange(8), jax.devices()[1])
    a + b  # Raises an error
    ```

    See https://jax.readthedocs.io/en/latest/faq.html#controlling-data-and-computation-placement-on-devices
    for more information.
    """

  @property
  @abc.abstractmethod
  def device(self) -> Device | Sharding:
    """Array API-compatible device attribute.

    For single-device arrays, this returns a Device. For sharded arrays, this
    returns a Sharding.
    """

  @abc.abstractmethod
  def copy_to_host_async(self):
    """Copies an ``Array`` to the host asynchronously.

    For arrays that live an an accelerator, such as a GPU or a TPU, JAX may
    cache the value of the array on the host. Normally this happens
    behind the scenes when the value of an on-device array is requested by the
    user, but waiting to initiate a device-to-host copy until the value is
    requested requires that JAX block the caller while waiting for the copy to
    complete.

    ``copy_to_host_async`` requests that JAX populate its on-host cache of an
    array, but does not wait for the copy to complete. This may speed up a
    future on-host access to the array's contents.
    """


Array.__module__ = "jax"

# StaticScalar is the Union of all scalar types that can be converted to
# JAX arrays, and are possible to mark as static arguments.
StaticScalar = Union[
  np.bool_, np.number,  # NumPy scalar types
  bool, int, float, complex,  # Python scalar types
]
StaticScalar.__doc__ = "Type annotation for JAX-compatible static scalars."


# ArrayLike is a Union of all objects that can be implicitly converted to a
# standard JAX array (i.e. not including future non-standard array types like
# KeyArray and BInt). It's different than np.typing.ArrayLike in that it doesn't
# accept arbitrary sequences, nor does it accept string data.
ArrayLike = Union[
  Array,  # JAX array type
  np.ndarray,  # NumPy array type
  StaticScalar,  # valid scalars
]
ArrayLike.__doc__ = "Type annotation for JAX array-like objects."
