# Copyright 2024 The Flax Authors.
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

"""Boxed Metadata API

Boxed metadata enables tracking arbitrary metadata for linen variables
that is compatible with lifted transformations.

See ``Partitioned`` for a practical example on how to use this metadata
to keep track of how variables should be partitioned with ``jax.pjit``.
"""

import abc
import dataclasses
import functools
from typing import Any, Generic, TypeVar
from collections.abc import Callable

from flax import errors, struct
from flax.typing import LogicalNames
import jax
from jax.interpreters import pxla

A = TypeVar('A')
B = TypeVar('B')
TAxisMetadata = TypeVar('TAxisMetadata', bound='AxisMetadata[Any]')


class AxisMetadata(Generic[A], metaclass=abc.ABCMeta):
  """Abstract base class for boxed Metadata.

  ``AxisMetadata`` enables arbitrary, per axis metadata for variables.
  By using ``unbox`` the metadata is stripped away to obtain the original
  variables. By using unboxing, most code handling variables does not need
  to handle ``AxisMetadata`` specifically, but can directly operate on the JAX
  arrays that they wrap.

  Additionally, ``AxisMetadata`` supports updating metadata whenever an axis
  is added or removed by a functional transformation
  (e.g.: ``nn.scan`` or ``nn.vmap``) using the ``add_axis`` and ``remove_axis``
  methods.

  By extending ``AxisMetadata``, custom metadata can be stored. See
  ``Partitioned`` for a specific implementation.
  """

  @abc.abstractmethod
  def unbox(self) -> A:
    """Returns the content of the AxisMetadata box.

    Note that unlike ``meta.unbox`` the unbox call should not recursively unbox
    metadata. It should simply return value that it wraps directly even
    if that value itself is an instance of AxisMetadata.

    In practise, AxisMetadata subclasses should be registered as PyTree nodes to
    support passing instances to JAX and Flax APIs. The leaves returned for this
    node should correspond to the value returned by unbox.

    Returns:
      The unboxed value.
    """
    pass

  @abc.abstractmethod
  def replace_boxed(self, val: B) -> 'AxisMetadata[B]':
    """Replaces the boxed value with the provided value.

    Args:
      val: The new value to be boxed by this AxisMetadata wrapper

    Returns:
      A new instance of the same type as self with `val` as the new ``unbox``
      content
    """
    pass

  @abc.abstractmethod
  def add_axis(
      self: TAxisMetadata, index: int, params: dict[Any, Any]
  ) -> TAxisMetadata:
    """Adds a new axis to the axis metadata.

    Note that add_axis and remove_axis should act as each other's inverse
    (meaning: ``x.add_axis(i, p).remove_axis(i, p) == x``)

    Args:
      index: The position at which the new axis will be inserted
      params: An arbitrary dictionary of parameters passed by the transformation
        that introduces the new axis (e.g.: ``nn.scan`` or ``nn.vmap``). The
        user passes this dictionary as the `metadata_param` argument to the
        transformation.

    Returns:
      A new instance of the same type as self and with the same ``unbox``
      content with updated axis metadata.
    """
    pass

  @abc.abstractmethod
  def remove_axis(
      self: TAxisMetadata, index: int, params: dict[Any, Any]
  ) -> TAxisMetadata:
    """Removes an axis from the axis metadata.

    Note that add_axis and remove_axis should act as each other's inverse
    (meaning: ``x.remove_axis(i, p).add_axis(i, p) == x``)

    Args:
      index: The position of the axis that is to be removed
      params: An arbitrary dictionary of parameters passed by the transformation
        that introduced the axis (e.g.: ``nn.scan`` or ``nn.vmap``). The user
        passes this dictionary as the `metadata_param` argument to the
        transformation.

    Returns:
      A new instance of the same type as self and with the same ``unbox``
      content with updated axis metadata.
    """
    pass


def is_axis_metadata(val: Any) -> bool:
  """Returns whether the argument is an instance of AxisMetadata."""
  return isinstance(val, AxisMetadata)


def map_axis_meta(fn: Callable[[AxisMetadata[Any]], Any], tree: Any) -> Any:
  """Maps over all PyTree nodes that are AxisMetadata instances."""

  def wrapper(x):
    if isinstance(x, AxisMetadata):
      return fn(x)
    else:
      return x

  return jax.tree_util.tree_map(wrapper, tree, is_leaf=is_axis_metadata)


def add_axis(tree: Any, index: int, params: dict[Any, Any]) -> Any:
  """Add an axis to each AxisMetadata node in a PyTree."""
  return map_axis_meta(lambda x: x.add_axis(index, params), tree)


def remove_axis(tree: Any, index: int, params: dict[Any, Any]) -> Any:
  """Remove an axis from each AxisMetadata node in a PyTree."""
  return map_axis_meta(lambda x: x.remove_axis(index, params), tree)


def unbox(tree: Any) -> Any:
  """Strips all AxisMetadata boxes from a PyTree."""
  return map_axis_meta(lambda x: unbox(x.unbox()), tree)


def replace_boxed(tree: Any, updates: Any) -> Any:
  """Updates all AxisMetadata boxes with the values in updates."""

  def inner_update(c, v):
    if isinstance(c, AxisMetadata):
      return c.replace_boxed(replace_boxed(c.unbox(), v))
    else:
      return v

  return jax.tree_util.tree_map(
      inner_update, tree, updates, is_leaf=is_axis_metadata
  )


PARTITION_NAME = 'partition_name'


def _global_mesh_defined() -> bool:
  """Checks if global mesh resource environment is defined."""
  env = pxla.thread_resources.env
  return env.physical_mesh.devices.shape != ()  # pylint: disable=g-explicit-bool-comparison


class Partitioned(struct.PyTreeNode, AxisMetadata[A]):
  """Wrapper for partitioning metadata.

  ``Partitioned`` is used to extend variables with partitioning information
  required for ``jax.experimental.pjit``.

  The easiest way to define Partitioned variables is by using the
  ``with_partitioning`` wrapper around the variable initializer.

  Example::

    class MLP(nn.Module):
      hidden_size: int
      @nn.compact
      def __call__(self, x):
        ki = nn.linear.default_kernel_init
        h = nn.Dense(
            self.hidden_size,
            kernel_init=nn.with_partitioning(ki, ('data', 'model')))(x)
        h = nn.relu(h)
        return nn.Dense(
            x.shape[-1],
            kernel_init=nn.with_partitioning(ki, ('model', 'data')))(h)
    mlp = MLP(4096)
    x = jnp.ones((8 * 1024, 1024))
    # use eval_shape to get the Partitioned instances for the variables.
    # this way we can determine the PartitionSpecs for the init variables
    # before we call the init fn.
    var_spec = nn.get_partition_spec(
        jax.eval_shape(mlp.init, random.key(0), x))
    init_fn = mesh(pjit(mlp.init,
                        (None, PartitionSpec("data", "model")), var_spec))
    variables = init_fn(random.key(0), x)
    apply_fn = mesh(pjit(
        mlp.apply,
        (var_spec, PartitionSpec("data", "model")),
         PartitionSpec("data", "model")))
    apply_fn(variables, x)


  ``Partitioned`` values can gain additional axes when using transformations
  like ``nn.vmap`` and ``nn.scan``. In this case you can specify the name of
  the new axis with the `metadata_params` args in vmap/scan::

    class Model(nn.Module):
    @nn.compact
    def __call__(self, x):
      def body(mdl, c):
        c = MLP(4096)(c)
        return c, ()
      c, _ = nn.scan(
          body, variable_axes={"params": 0}, split_rngs={"params": 0}, length=8,
          metadata_params={nn.meta.PARTITION_NAME: "layers"})(self, x)
      return c
  """

  value: Any
  names: LogicalNames = struct.field(pytree_node=False)
  mesh: jax.sharding.Mesh | None = struct.field(
      default=None, pytree_node=False
  )

  def unbox(self, apply_constraint=True) -> A:
    """Returns the wrapped value with the partitioning applied as a sharding constraint."""
    if apply_constraint and (_global_mesh_defined() or self.mesh is not None):
      axis_resource = self.get_partition_spec()
      if self.mesh is not None:
        sharding = jax.sharding.NamedSharding(self.mesh, axis_resource)
        return jax.lax.with_sharding_constraint(self.value, sharding)
      return jax.lax.with_sharding_constraint(self.value, axis_resource)
    else:
      return self.value

  def replace_boxed(self, val: B) -> 'Partitioned[B]':
    return self.replace(value=val)  # type: ignore

  def _get_partition_name(self, params: dict[Any, Any]) -> str:
    if PARTITION_NAME not in params:
      raise errors.PartitioningUnspecifiedError(self)
    return params[PARTITION_NAME]

  def add_axis(self, index: int, params: dict[Any, Any]) -> 'Partitioned[A]':
    axis_name = self._get_partition_name(params)
    names = list(self.names)
    while len(names) < index:
      names.append(None)  # type: ignore
    names.insert(index, axis_name)  # type: ignore
    return self.replace(names=tuple(names))

  def remove_axis(self, index: int, params: dict[Any, Any]) -> 'Partitioned[A]':
    axis_name = self._get_partition_name(params)
    names = list(self.names)
    assert names.pop(index) == axis_name
    return self.replace(names=tuple(names))

  def get_partition_spec(self) -> jax.sharding.PartitionSpec:
    """Returns the ``Partitionspec`` for this partitioned value."""
    return jax.sharding.PartitionSpec(*self.names)

  def get_sharding(self, mesh: jax.sharding.Mesh) -> jax.sharding.Sharding:
    """Returns the ``NamedSharding`` for this partitioned value."""
    return jax.sharding.NamedSharding(mesh, self.get_partition_spec())

  def to_nnx_metadata(self) -> dict[str, Any]:
    """Return a dict of metadata that can translate into an `nnx.Variable`."""
    metadata = vars(self)
    metadata['sharding'] = metadata.pop('names')
    return metadata

  @classmethod
  def from_nnx_metadata(cls, metadata: dict[str, Any]):
    """Given a dict of `nnx.Variable` format metadata, create a `nn.Partitioned`."""
    metadata['names'] = metadata.pop('sharding')
    fields = {x.name for x in dataclasses.fields(cls)}
    return cls(**{k: v for k, v in metadata.items() if k in fields})


def with_partitioning(
    fn: Callable[..., Any],
    names: LogicalNames,
    mesh: jax.sharding.Mesh | None = None,
) -> Callable[..., Partitioned[Any]]:
  """Wraps a function's return value with Partitioned.

  Example::

    >>> import flax.linen as nn
    >>> kernel_init = nn.with_partitioning(
    ...     nn.initializers.lecun_normal(), (None, "data"))
    >>> partitioned_dense = nn.Dense(features=3, kernel_init=kernel_init)

  Args:
    fn: The function to be wrapped. Typically this is an initializer.
    names: The logical axis passed to ``Partitioned``.
    mesh: The mesh to use for the partitioning. If None, the global mesh
      resource is used if available.

  Returns:
    A function wrapping ``fn`` that will return an instance of ``Partitioned``.
  """

  @functools.wraps(fn)
  def wrapper(*args, **kwargs):
    return Partitioned(fn(*args, **kwargs), names, mesh=mesh)

  return wrapper


def _get_leaf_pspec(x: Any) -> jax.sharding.PartitionSpec | None:
  if hasattr(x, 'get_partition_spec'):
    return x.get_partition_spec()
  # Unboxed arrays, which should be replicated across all devices
  elif hasattr(x, 'shape'):
    return jax.sharding.PartitionSpec()
  else:
    return None


def get_partition_spec(tree: Any) -> Any:
  """Extracts a PartitionSpec tree from a PyTree containing ``Partitioned`` values."""
  return jax.tree_util.tree_map(
      _get_leaf_pspec, tree, is_leaf=lambda x: isinstance(x, AxisMetadata)
  )


def get_sharding(tree: Any, mesh: jax.sharding.Mesh) -> Any:
  """Extracts a jax.sharding tree from a PyTree containing ``Partitioned`` values and a mesh."""
  def f(x: Any) -> jax.sharding.Sharding | None:
    if hasattr(x, 'get_sharding'):
      return x.get_sharding(mesh)
    pspec = _get_leaf_pspec(x)
    if pspec is None:
      return None
    return jax.sharding.NamedSharding(mesh, pspec)

  return jax.tree_util.tree_map(
      f, tree, is_leaf=lambda x: isinstance(x, AxisMetadata)
  )
