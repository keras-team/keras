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

"""Utilities for defining custom classes that can be used with jax transformations."""

from collections.abc import Callable
import dataclasses
import functools
from typing import TypeVar, overload

import jax
from typing_extensions import (
  dataclass_transform,  # pytype: disable=not-supported-yet
)

from . import serialization

_T = TypeVar('_T')


def field(pytree_node=True, *, metadata=None, **kwargs):
  return dataclasses.field(metadata=(metadata or {}) | {'pytree_node': pytree_node},
                           **kwargs)


@dataclass_transform(field_specifiers=(field,))  # type: ignore[literal-required]
@overload
def dataclass(clz: _T, **kwargs) -> _T:
  ...


@dataclass_transform(field_specifiers=(field,))  # type: ignore[literal-required]
@overload
def dataclass(**kwargs) -> Callable[[_T], _T]:
  ...


@dataclass_transform(field_specifiers=(field,))  # type: ignore[literal-required]
def dataclass(
    clz: _T | None = None,
    **kwargs,
) -> _T | Callable[[_T], _T]:
  """Create a class which can be passed to functional transformations.

  .. note::
    Inherit from ``PyTreeNode`` instead to avoid type checking issues when
    using PyType.

  Jax transformations such as ``jax.jit`` and ``jax.grad`` require objects that are
  immutable and can be mapped over using the ``jax.tree_util`` methods.
  The ``dataclass`` decorator makes it easy to define custom classes that can be
  passed safely to Jax. For example::

    >>> from flax import struct
    >>> import jax
    >>> from typing import Any, Callable

    >>> @struct.dataclass
    ... class Model:
    ...   params: Any
    ...   # use pytree_node=False to indicate an attribute should not be touched
    ...   # by Jax transformations.
    ...   apply_fn: Callable = struct.field(pytree_node=False)

    ...   def __apply__(self, *args):
    ...     return self.apply_fn(*args)

    >>> params = {}
    >>> params_b = {}
    >>> apply_fn = lambda v, x: x
    >>> model = Model(params, apply_fn)

    >>> # model.params = params_b  # Model is immutable. This will raise an error.
    >>> model_b = model.replace(params=params_b)  # Use the replace method instead.

    >>> # This class can now be used safely in Jax to compute gradients w.r.t. the
    >>> # parameters.
    >>> model = Model(params, apply_fn)
    >>> loss_fn = lambda model: 3.
    >>> model_grad = jax.grad(loss_fn)(model)

  Note that dataclasses have an auto-generated ``__init__`` where
  the arguments of the constructor and the attributes of the created
  instance match 1:1. This correspondence is what makes these objects
  valid containers that work with JAX transformations and
  more generally the ``jax.tree_util`` library.

  Sometimes a "smart constructor" is desired, for example because
  some of the attributes can be (optionally) derived from others.
  The way to do this with Flax dataclasses is to make a static or
  class method that provides the smart constructor.
  This way the simple constructor used by ``jax.tree_util`` is
  preserved. Consider the following example::

    >>> @struct.dataclass
    ... class DirectionAndScaleKernel:
    ...   direction: jax.Array
    ...   scale: jax.Array

    ...   @classmethod
    ...   def create(cls, kernel):
    ...     scale = jax.numpy.linalg.norm(kernel, axis=0, keepdims=True)
    ...     direction = direction / scale
    ...     return cls(direction, scale)

  Args:
    clz: the class that will be transformed by the decorator.
    **kwargs: arguments to pass to the dataclass constructor.

  Returns:
    The new class.
  """
  # Support passing arguments to the decorator (e.g. @dataclass(kw_only=True))
  if clz is None:
    return functools.partial(dataclass, **kwargs)  # type: ignore[bad-return-type]

  # check if already a flax dataclass
  if '_flax_dataclass' in clz.__dict__:
    return clz

  if 'frozen' not in kwargs.keys():
    kwargs['frozen'] = True
  data_clz = dataclasses.dataclass(**kwargs)(clz)  # type: ignore
  meta_fields = []
  data_fields = []
  for field_info in dataclasses.fields(data_clz):
    is_pytree_node = field_info.metadata.get('pytree_node', True)
    if is_pytree_node:
      data_fields.append(field_info.name)
    else:
      meta_fields.append(field_info.name)

  def replace(self, **updates):
    """Returns a new object replacing the specified fields with new values."""
    return dataclasses.replace(self, **updates)

  data_clz.replace = replace

  jax.tree_util.register_dataclass(data_clz, data_fields, meta_fields)

  def to_state_dict(x):
    state_dict = {
      name: serialization.to_state_dict(getattr(x, name))
      for name in data_fields
    }
    return state_dict

  def from_state_dict(x, state):
    """Restore the state of a data class."""
    state = state.copy()  # copy the state so we can pop the restored fields.
    updates = {}
    for name in data_fields:
      if name not in state:
        raise ValueError(
          f'Missing field {name} in state dict while restoring'
          f' an instance of {clz.__name__},'
          f' at path {serialization.current_path()}'
        )
      value = getattr(x, name)
      value_state = state.pop(name)
      updates[name] = serialization.from_state_dict(
        value, value_state, name=name
      )
    if state:
      names = ','.join(state.keys())
      raise ValueError(
        f'Unknown field(s) "{names}" in state dict while'
        f' restoring an instance of {clz.__name__}'
        f' at path {serialization.current_path()}'
      )
    return x.replace(**updates)

  serialization.register_serialization_state(
    data_clz, to_state_dict, from_state_dict
  )

  # add a _flax_dataclass flag to distinguish from regular dataclasses
  data_clz._flax_dataclass = True  # type: ignore[attr-defined]

  return data_clz  # type: ignore


TNode = TypeVar('TNode', bound='PyTreeNode')


@dataclass_transform(field_specifiers=(field,))  # type: ignore[literal-required]
class PyTreeNode:
  """Base class for dataclasses that should act like a JAX pytree node.

  See ``flax.struct.dataclass`` for the ``jax.tree_util`` behavior.
  This base class additionally avoids type checking errors when using PyType.

  Example::

    >>> from flax import struct
    >>> import jax
    >>> from typing import Any, Callable

    >>> class Model(struct.PyTreeNode):
    ...   params: Any
    ...   # use pytree_node=False to indicate an attribute should not be touched
    ...   # by Jax transformations.
    ...   apply_fn: Callable = struct.field(pytree_node=False)

    ...   def __apply__(self, *args):
    ...     return self.apply_fn(*args)

    >>> params = {}
    >>> params_b = {}
    >>> apply_fn = lambda v, x: x
    >>> model = Model(params, apply_fn)

    >>> # model.params = params_b  # Model is immutable. This will raise an error.
    >>> model_b = model.replace(params=params_b)  # Use the replace method instead.

    >>> # This class can now be used safely in Jax to compute gradients w.r.t. the
    >>> # parameters.
    >>> model = Model(params, apply_fn)
    >>> loss_fn = lambda model: 3.
    >>> model_grad = jax.grad(loss_fn)(model)
  """

  def __init_subclass__(cls, **kwargs):
    dataclass(cls, **kwargs)  # pytype: disable=wrong-arg-types

  def __init__(self, *args, **kwargs):
    # stub for pytype
    raise NotImplementedError

  def replace(self: TNode, **overrides) -> TNode:
    # stub for pytype
    raise NotImplementedError
