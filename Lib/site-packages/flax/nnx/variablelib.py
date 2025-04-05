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
# pytype: skip-file
from __future__ import annotations

import dataclasses
import functools
from functools import partial
import typing as tp
from typing import Any

import jax
import treescope  # type: ignore[import-untyped]

from flax import errors
from flax.nnx import filterlib, reprlib, tracers, visualization
from flax.typing import Missing, PathParts, SizeBytes
import jax.tree_util as jtu

A = tp.TypeVar('A')
B = tp.TypeVar('B')
F = tp.TypeVar('F', bound=tp.Callable[..., tp.Any])
V = tp.TypeVar('V', bound='Variable[Any]')
GetValueHook = tp.Callable[['Variable[A]', A], A]
SetValueHook = tp.Callable[['Variable[A]', A], A]
CreateValueHook = tp.Callable[['Variable[A]', A], A]
AxisName = str
AxisIndex = int
AddAxisHook = tp.Callable[[V, AxisIndex, AxisName | None], None]
RemoveAxisHook = tp.Callable[[V, AxisIndex, AxisName | None], None]


@dataclasses.dataclass
class VariableMetadata(tp.Generic[A]):
  raw_value: A
  set_value_hooks: tuple[SetValueHook[A], ...] = ()
  get_value_hooks: tuple[GetValueHook[A], ...] = ()
  create_value_hooks: tuple[CreateValueHook[A], ...] = ()
  add_axis_hooks: tuple[AddAxisHook[Variable[A]], ...] = ()
  remove_axis_hooks: tuple[RemoveAxisHook[Variable[A]], ...] = ()
  metadata: tp.Mapping[str, tp.Any] = dataclasses.field(default_factory=dict)


class Variable(tp.Generic[A], reprlib.Representable):
  """The base class for all ``Variable`` types. Create custom ``Variable``
  types by subclassing this class. Numerous NNX graph functions can filter
  for specific ``Variable`` types, for example, :func:`split`, :func:`state`,
  :func:`pop`, and :func:`State.filter`.

  Example usage::

    >>> from flax import nnx
    >>> import jax, jax.numpy as jnp

    >>> class CustomVariable(nnx.Variable):
    ...   pass

    >>> class Model(nnx.Module):
    ...   def __init__(self, rngs):
    ...     self.linear = nnx.Linear(2, 3, rngs=rngs)
    ...     self.custom_variable = CustomVariable(jnp.ones((1, 3)))
    ...   def __call__(self, x):
    ...     return self.linear(x) + self.custom_variable
    >>> model = Model(rngs=nnx.Rngs(0))

    >>> linear_variables = nnx.state(model, nnx.Param)
    >>> jax.tree.map(jnp.shape, linear_variables)
    State({
      'linear': {
        'bias': VariableState(
          type=Param,
          value=(3,)
        ),
        'kernel': VariableState(
          type=Param,
          value=(2, 3)
        )
      }
    })

    >>> custom_variable = nnx.state(model, CustomVariable)
    >>> jax.tree.map(jnp.shape, custom_variable)
    State({
      'custom_variable': VariableState(
        type=CustomVariable,
        value=(1, 3)
      )
    })

    >>> variables = nnx.state(model)
    >>> jax.tree.map(jnp.shape, variables)
    State({
      'custom_variable': VariableState(
        type=CustomVariable,
        value=(1, 3)
      ),
      'linear': {
        'bias': VariableState(
          type=Param,
          value=(3,)
        ),
        'kernel': VariableState(
          type=Param,
          value=(2, 3)
        )
      }
    })
  """

  __slots__ = ('raw_value', '_trace_state', '_var_metadata')

  raw_value: A
  _trace_state: tracers.TraceState
  _var_metadata: dict[str, tp.Any]

  def __init__(
    self,
    value: tp.Union[A, VariableMetadata[A]],
    **metadata: tp.Any,
  ):
    var_t = type(self)
    object.__setattr__(self, '_trace_state', tracers.TraceState())

    if isinstance(value, VariableMetadata):
      metadata.update(value.metadata)
      value = tp.cast(A, value.raw_value)

    object.__setattr__(self, 'raw_value', value)

    if hasattr(var_t, 'on_get_value') and 'on_get_value' not in metadata:
      metadata['on_get_value'] = var_t.on_get_value

    if hasattr(var_t, 'on_set_value') and 'on_set_value' not in metadata:
      metadata['on_set_value'] = var_t.on_set_value

    if hasattr(var_t, 'on_create_value') and 'on_create_value' not in metadata:
      metadata['on_create_value'] = var_t.on_create_value

    if hasattr(var_t, 'on_add_axis') and 'on_add_axis' not in metadata:
      metadata['on_add_axis'] = var_t.on_add_axis

    if hasattr(var_t, 'on_remove_axis') and 'on_remove_axis' not in metadata:
      metadata['on_remove_axis'] = var_t.on_remove_axis

    object.__setattr__(self, '_var_metadata', metadata)
    # run create_value hooks
    object.__setattr__(self, 'raw_value', self.create_value(self.raw_value))

  def __getattr__(self, name: str) -> tp.Any:
    if name in object.__getattribute__(self, '_var_metadata'):
      return self._var_metadata[name]
    return getattr(self.value, name)

  def __setattr__(self, name: str, value: tp.Any):
    if not self._trace_state.is_valid():
      raise errors.TraceContextError(
        f'Cannot mutate {type(self).__name__} from a different trace level'
      )

    if (
      name == 'value'
      or name == 'raw_value'
      or name == '_var_metadata'
      or name == '_trace_state'
    ):
      object.__setattr__(self, name, value)
    else:
      self._var_metadata[name] = value

  def __delattr__(self, name: str):
    if not self._trace_state.is_valid():
      raise errors.TraceContextError(
        f'Cannot mutate {type(self).__name__} from a different trace level'
      )

    if (
      name == 'value'
      or name == 'raw_value'
      or name == '_var_metadata'
      or name == '_trace_state'
    ):
      object.__delattr__(self, name)
    else:
      del self._var_metadata[name]

  @classmethod
  def state(cls, value: A, **metadata) -> VariableState[A]:
    return cls(value, **metadata).to_state()

  def get_metadata(self):
    return self._var_metadata

  def copy_from(self, other: Variable[A]) -> None:
    if type(self) is not type(other):
      raise ValueError(
        f'Cannot copy from incompatible container, '
        f'expected {type(self).__name__}, got {type(other).__name__}'
      )
    if self is other:
      return
    self.raw_value = other.raw_value
    self._var_metadata.clear()
    self._var_metadata.update(other.get_metadata())

  def update_from_state(self, variable_state: VariableState[A]):
    object.__setattr__(self, 'raw_value', variable_state.value)
    object.__setattr__(
        self, '_var_metadata', variable_state._var_metadata.copy()
    )

  @property
  def value(self) -> A:
    value = self.raw_value
    if 'on_get_value' in self._var_metadata:
      value = self._var_metadata['on_get_value'](self, value)
    return value

  @value.setter
  def value(self, value: A):
    if isinstance(value, Variable):
      raise ValueError(
        'Cannot set value to a Variable, ' 'use `copy_from` method instead'
      )
    if 'on_set_value' in self._var_metadata:
      value = self._var_metadata['on_set_value'](self, value)
    object.__setattr__(self, 'raw_value', value)

  def create_value(self, value: A):
    if 'on_create_value' in self._var_metadata:
      value = self._var_metadata['on_create_value'](self, value)
    return value

  def add_axis(self, axis_index: AxisIndex, axis_name: AxisName | None):
    if 'on_add_axis' in self._var_metadata:
      self._var_metadata['on_add_axis'](self, axis_index, axis_name)

  def remove_axis(self, axis_index: AxisIndex, axis_name: AxisName | None):
    if 'on_remove_axis' in self._var_metadata:
      self._var_metadata['on_remove_axis'](self, axis_index, axis_name)

  @tp.overload
  def replace(self, value: B, **kwargs) -> Variable[B]:
    ...

  @tp.overload
  def replace(self, **kwargs) -> Variable[A]:
    ...

  def replace(self, value: tp.Any = Missing, **kwargs) -> Variable[tp.Any]:
    if value is not Missing:
      kwargs['raw_value'] = value

    # rename `value` to `raw_value`
    if 'value' in kwargs:
      kwargs['raw_value'] = kwargs.pop('value')

    # return `value` if it is a Variable
    if 'raw_value' in kwargs and isinstance(
      value := kwargs['raw_value'], Variable
    ):
      # remove value from kwargs
      kwargs.pop('raw_value')
      if type(self) is not type(value):
        raise ValueError(
          'Cannot replace value from incompatible container, '
          f'expected {type(self).__name__}, got {type(value).__name__}'
        )
      # if kwargs aren't empty, recursively call replace
      # else return variable value
      if kwargs:
        return value.replace(**kwargs)
      else:
        return value

    # get and update attributes
    # return new instance with updated attributes
    obj = object.__new__(type(self))
    object.__setattr__(obj, '_trace_state', self._trace_state)
    object.__setattr__(obj, 'raw_value', kwargs.pop('raw_value'))
    object.__setattr__(obj, '_var_metadata', self.get_metadata())
    obj._var_metadata.update(kwargs)
    return obj

  @classmethod
  def from_metadata(cls, value: A, attributes: dict[str, tp.Any]):
    obj = object.__new__(cls)
    object.__setattr__(obj, '_trace_state', tracers.TraceState())
    object.__setattr__(obj, 'raw_value', value)
    object.__setattr__(obj, '_var_metadata', attributes)
    return obj

  def copy(self: Variable[A]) -> Variable[A]:
    obj = object.__new__(type(self))
    object.__setattr__(obj, '_trace_state', self._trace_state)
    object.__setattr__(obj, 'raw_value', self.raw_value)
    object.__setattr__(obj, '_var_metadata', self.get_metadata().copy())
    return obj

  def to_state(self: Variable[A]) -> VariableState[A]:
    return VariableState(type(self), self.raw_value, **self._var_metadata)

  def __nnx_repr__(self):
    stats = SizeBytes.from_any(self.value)
    if stats:
      comment = f' # {stats}'
    else:
      comment = ''

    yield reprlib.Object(type=type(self).__name__, comment=comment)
    yield reprlib.Attr('value', self.raw_value)
    for name, value in self._var_metadata.items():
      yield reprlib.Attr(name, repr(value))

  def __treescope_repr__(self, path, subtree_renderer):
    size_bytes = SizeBytes.from_any(self.value)
    if size_bytes:
      stats_repr = f' # {size_bytes}'
      first_line_annotation = treescope.rendering_parts.comment_color(
        treescope.rendering_parts.text(f'{stats_repr}')
      )
    else:
      first_line_annotation = None

    children = {'value': self.raw_value, **self._var_metadata}
    return visualization.render_object_constructor(
      object_type=type(self),
      attributes=children,
      path=path,
      subtree_renderer=subtree_renderer,
      first_line_annotation=first_line_annotation,
    )

  # hooks API
  if tp.TYPE_CHECKING:

    def on_get_value(self, value: A) -> A: ...

    def on_set_value(self, value: A) -> A: ...

    def on_create_value(self, value: A) -> A: ...

    def on_add_axis(
      self: V, axis_index: AxisIndex, axis_name: AxisName | None
    ) -> V: ...

    def on_remove_axis(
      self: V, axis_index: AxisIndex, axis_name: AxisName | None
    ) -> V: ...

  def __jax_array__(self):
    return self.value

  # pickle support
  def __getstate__(self):
    return {
        'raw_value': self.raw_value,
        '_trace_state': self._trace_state,
        '_var_metadata': self._var_metadata,
    }

  def __setstate__(self, state):
    object.__setattr__(self, 'raw_value', state['raw_value'])
    object.__setattr__(self, '_trace_state', state['_trace_state'])
    object.__setattr__(self, '_var_metadata', state['_var_metadata'])

  # --------------------------------------------
  # proxy methods
  # --------------------------------------------

  def __getitem__(self, key) -> tp.Any:
    return self.value[key]  # type: ignore

  def __setitem__(self, key, value) -> None:
    self.value[key] = value  # type: ignore

  def __call__(self, *args, **kwargs) -> tp.Any:
    return self.value(*args, **kwargs)  # type: ignore

  def __len__(self) -> int:
    return len(self.value)  # type: ignore

  def __iter__(self) -> tp.Iterator:
    return iter(self.value)  # type: ignore

  def __contains__(self, item) -> bool:
    return item in self.value  # type: ignore

  def __add__(self, other) -> A:
    return self.value.__add__(other)  # type: ignore

  def __sub__(self, other) -> A:
    return self.value.__sub__(other)  # type: ignore

  def __mul__(self, other) -> A:
    return self.value.__mul__(other)  # type: ignore

  def __matmul__(self, other) -> A:
    return self.value.__matmul__(other)  # type: ignore

  def __truediv__(self, other) -> A:
    return self.value.__truediv__(other)  # type: ignore

  def __floordiv__(self, other) -> A:
    return self.value.__floordiv__(other)  # type: ignore

  def __mod__(self, other) -> A:
    return self.value.__mod__(other)  # type: ignore

  def __divmod__(self, other) -> A:
    return self.value.__divmod__(other)  # type: ignore

  def __pow__(self, other) -> A:
    return self.value.__pow__(other)  # type: ignore

  def __lshift__(self, other) -> A:
    return self.value.__lshift__(other)  # type: ignore

  def __rshift__(self, other) -> A:
    return self.value.__rshift__(other)  # type: ignore

  def __and__(self, other) -> A:
    return self.value.__and__(other)  # type: ignore

  def __xor__(self, other) -> A:
    return self.value.__xor__(other)  # type: ignore

  def __or__(self, other) -> A:
    return self.value.__or__(other)  # type: ignore

  def __radd__(self, other) -> A:
    return self.value.__radd__(other)  # type: ignore

  def __rsub__(self, other) -> A:
    return self.value.__rsub__(other)  # type: ignore

  def __rmul__(self, other) -> A:
    return self.value.__rmul__(other)  # type: ignore

  def __rmatmul__(self, other) -> A:
    return self.value.__rmatmul__(other)  # type: ignore

  def __rtruediv__(self, other) -> A:
    return self.value.__rtruediv__(other)  # type: ignore

  def __rfloordiv__(self, other) -> A:
    return self.value.__rfloordiv__(other)  # type: ignore

  def __rmod__(self, other) -> A:
    return self.value.__rmod__(other)  # type: ignore

  def __rdivmod__(self, other) -> A:
    return self.value.__rdivmod__(other)  # type: ignore

  def __rpow__(self, other) -> A:
    return self.value.__rpow__(other)  # type: ignore

  def __rlshift__(self, other) -> A:
    return self.value.__rlshift__(other)  # type: ignore

  def __rrshift__(self, other) -> A:
    return self.value.__rrshift__(other)  # type: ignore

  def __rand__(self, other) -> A:
    return self.value.__rand__(other)  # type: ignore

  def __rxor__(self, other) -> A:
    return self.value.__rxor__(other)  # type: ignore

  def __ror__(self, other) -> A:
    return self.value.__ror__(other)  # type: ignore

  def __iadd__(self: V, other) -> V:
    value = self.value
    if hasattr(value, '__iadd__'):
      value.__iadd__(other)
    else:
      self.value = value.__add__(other)
    return self

  def __isub__(self: V, other) -> V:
    value = self.value
    if hasattr(value, '__isub__'):
      value.__isub__(other)
    else:
      self.value = value.__sub__(other)
    return self

  def __imul__(self: V, other) -> V:
    value = self.value
    if hasattr(value, '__imul__'):
      value.__imul__(other)
    else:
      self.value = value.__mul__(other)
    return self

  def __imatmul__(self: V, other) -> V:
    value = self.value
    if hasattr(value, '__imatmul__'):
      value.__imatmul__(other)
    else:
      self.value = value.__matmul__(other)
    return self

  def __itruediv__(self: V, other) -> V:
    value = self.value
    if hasattr(value, '__itruediv__'):
      value.__itruediv__(other)
    else:
      self.value = value.__truediv__(other)
    return self

  def __ifloordiv__(self: V, other) -> V:
    value = self.value
    if hasattr(value, '__ifloordiv__'):
      value.__ifloordiv__(other)
    else:
      self.value = value.__floordiv__(other)
    return self

  def __imod__(self: V, other) -> V:
    value = self.value
    if hasattr(value, '__imod__'):
      value.__imod__(other)
    else:
      self.value = value.__mod__(other)
    return self

  def __ipow__(self: V, other) -> V:
    value = self.value
    if hasattr(value, '__ipow__'):
      value.__ipow__(other)
    else:
      self.value = value.__pow__(other)
    return self

  def __ilshift__(self: V, other) -> V:
    value = self.value
    if hasattr(value, '__ilshift__'):
      value.__ilshift__(other)
    else:
      self.value = value.__lshift__(other)
    return self

  def __irshift__(self: V, other) -> V:
    value = self.value
    if hasattr(value, '__irshift__'):
      value.__irshift__(other)
    else:
      self.value = value.__rshift__(other)
    return self

  def __iand__(self: V, other) -> V:
    value = self.value
    if hasattr(value, '__iand__'):
      value.__iand__(other)
    else:
      self.value = value.__and__(other)
    return self

  def __ixor__(self: V, other) -> V:
    value = self.value
    if hasattr(value, '__ixor__'):
      value.__ixor__(other)
    else:
      self.value = value.__xor__(other)
    return self

  def __ior__(self: V, other) -> V:
    value = self.value
    if hasattr(value, '__ior__'):
      value.__ior__(other)
    else:
      self.value = value.__or__(other)
    return self

  def __neg__(self) -> A:
    return self.value.__neg__()  # type: ignore

  def __pos__(self) -> A:
    return self.value.__pos__()  # type: ignore

  def __abs__(self) -> A:
    return self.value.__abs__()  # type: ignore

  def __invert__(self) -> A:
    return self.value.__invert__()  # type: ignore

  def __complex__(self) -> A:
    return self.value.__complex__()  # type: ignore

  def __int__(self) -> A:
    return self.value.__int__()  # type: ignore

  def __float__(self) -> A:
    return self.value.__float__()  # type: ignore

  def __index__(self) -> A:
    return self.value.__index__()  # type: ignore

  def __round__(self, ndigits: int) -> A:
    return self.value.__round__(ndigits)  # type: ignore

  def __trunc__(self) -> A:
    return self.value.__trunc__()  # type: ignore

  def __floor__(self) -> A:
    return self.value.__floor__()  # type: ignore

  def __ceil__(self) -> A:
    return self.value.__ceil__()  # type: ignore


class Param(Variable[A]):
  """The canonical learnable parameter. All learnable parameters
  in NNX layer modules will have the ``Param`` :class:`Variable`
  type::

    >>> from flax import nnx
    >>> import jax, jax.numpy as jnp

    >>> layer = nnx.Linear(2, 3, rngs=nnx.Rngs(0))
    >>> jax.tree.map(jnp.shape, nnx.state(layer))
    State({
      'bias': VariableState(
        type=Param,
        value=(3,)
      ),
      'kernel': VariableState(
        type=Param,
        value=(2, 3)
      )
    })
  """

  pass


class BatchStat(Variable[A]):
  """The mean and variance batch statistics stored in
  the :class:`BatchNorm` layer. Note, these are not the
  learnable scale and bias parameters, but rather the
  running average statistics that are typically used
  during post-training inference::

    >>> from flax import nnx
    >>> import jax, jax.numpy as jnp

    >>> layer = nnx.BatchNorm(3, rngs=nnx.Rngs(0))
    >>> jax.tree.map(jnp.shape, nnx.state(layer))
    State({
      'bias': VariableState(
        type=Param,
        value=(3,)
      ),
      'mean': VariableState(
        type=BatchStat,
        value=(3,)
      ),
      'scale': VariableState(
        type=Param,
        value=(3,)
      ),
      'var': VariableState(
        type=BatchStat,
        value=(3,)
      )
    })
  """

  pass


class Cache(Variable[A]):
  """Autoregressive cache in :class:`MultiHeadAttention`::

  >>> from flax import nnx
  >>> import jax, jax.numpy as jnp

  >>> layer = nnx.MultiHeadAttention(
  ...   num_heads=2,
  ...   in_features=3,
  ...   qkv_features=6,
  ...   out_features=6,
  ...   decode=True,
  ...   rngs=nnx.Rngs(0),
  ... )

  >>> layer.init_cache((1, 3))
  >>> jax.tree.map(jnp.shape, nnx.state(layer, nnx.Cache))
  State({
    'cache_index': VariableState(
      type=Cache,
      value=()
    ),
    'cached_key': VariableState(
      type=Cache,
      value=(1, 2, 3)
    ),
    'cached_value': VariableState(
      type=Cache,
      value=(1, 2, 3)
    )
  })
  """

  pass


class Intermediate(Variable[A]):
  """:class:`Variable` type that is typically used for
  :func:`Module.sow`::

    >>> from flax import nnx
    >>> import jax, jax.numpy as jnp

    >>> class Model(nnx.Module):
    ...   def __init__(self, rngs):
    ...     self.linear1 = nnx.Linear(2, 3, rngs=rngs)
    ...     self.linear2 = nnx.Linear(3, 4, rngs=rngs)
    ...   def __call__(self, x):
    ...     x = self.linear1(x)
    ...     self.sow(nnx.Intermediate, 'i', x)
    ...     x = self.linear2(x)
    ...     return x
    >>> model = Model(rngs=nnx.Rngs(0))

    >>> x = jnp.ones((1, 2))
    >>> y = model(x)
    >>> jax.tree.map(jnp.shape, nnx.state(model, nnx.Intermediate))
    State({
      'i': VariableState(
        type=Intermediate,
        value=((1, 3),)
      )
    })
  """

  pass


class Perturbation(Intermediate[A]):
  """:class:`Variable` type that is typically used for
  :func:`Module.perturb`::

    >>> from flax import nnx
    >>> import jax, jax.numpy as jnp

    >>> class Model(nnx.Module):
    ...   def __init__(self, rngs):
    ...     self.linear1 = nnx.Linear(2, 3, rngs=rngs)
    ...     self.linear2 = nnx.Linear(3, 4, rngs=rngs)
    ...   def __call__(self, x):
    ...     x = self.linear1(x)
    ...     x = self.perturb('i', x)
    ...     x = self.linear2(x)
    ...     return x
    >>> model = Model(rngs=nnx.Rngs(0))

    >>> x = jnp.ones((1, 2))
    >>> y = model(x)
    >>> jax.tree.map(jnp.shape, nnx.state(model, nnx.Perturbation))
    State({
      'i': VariableState(
        type=Perturbation,
        value=(1, 3)
      )
    })
  """

  pass


class VariableState(tp.Generic[A], reprlib.Representable):
  __slots__ = ('type', 'value', '_var_metadata')
  type: type[Variable[A]]
  value: A
  _var_metadata: dict[str, tp.Any]

  def __init__(
    self,
    type: type[Variable[A]],  # type: ignore [valid-type]
    value: A,
    **metadata,
  ):
    object.__setattr__(self, 'type', type)
    object.__setattr__(self, 'value', value)
    object.__setattr__(self, '_var_metadata', metadata)

  def __getattr__(self, name: str) -> None:
    var_metadata = object.__getattribute__(self, '_var_metadata')
    if name not in var_metadata:
      raise AttributeError(f"'VariableState' object has no attribute '{name}'")
    return var_metadata[name]

  def __setattr__(self, name: str, value: Any) -> None:
    if name == 'type' or name == 'value' or name == '_var_metadata':
      object.__setattr__(self, name, value)
    else:
      self._var_metadata[name] = value

  def __delattr__(self, name: str) -> None:
    if name == 'type' or name == 'value' or name == '_var_metadata':
      object.__delattr__(self, name)
    else:
      del self._var_metadata[name]

  def __nnx_repr__(self):
    stats = SizeBytes.from_any(self.value)
    if stats:
      comment = f' # {stats}'
    else:
      comment = ''

    yield reprlib.Object(type=type(self), comment=comment)
    yield reprlib.Attr('type', self.type)
    yield reprlib.Attr('value', self.value)

    for name, value in self._var_metadata.items():
      yield reprlib.Attr(name, value)

  def __treescope_repr__(self, path, subtree_renderer):
    size_bytes = SizeBytes.from_any(self.value)
    if size_bytes:
      stats_repr = f' # {size_bytes}'
      first_line_annotation = treescope.rendering_parts.comment_color(
        treescope.rendering_parts.text(f'{stats_repr}')
      )
    else:
      first_line_annotation = None
    children = {'type': self.type, 'value': self.value, **self._var_metadata}
    return visualization.render_object_constructor(
      object_type=type(self),
      attributes=children,
      path=path,
      subtree_renderer=subtree_renderer,
      first_line_annotation=first_line_annotation,
    )

  def replace(self, value: B) -> VariableState[B]:
    return VariableState(self.type, value, **self.get_metadata())

  def to_variable(self) -> Variable[A]:
    # we use object.__new__ to avoid calling __init__ and bypass the
    # __init__ logic which should not be called twice
    variable = object.__new__(self.type)
    object.__setattr__(variable, '_trace_state', tracers.TraceState())
    object.__setattr__(variable, 'raw_value', self.value)
    object.__setattr__(variable, '_var_metadata', self.get_metadata().copy())
    return variable

  def copy(self: VariableState[A]) -> VariableState[A]:
    return jax.tree.map(lambda x: x, self)

  def get_metadata(self) -> dict[str, tp.Any]:
    return self._var_metadata

  def add_axis(self, axis_index: AxisIndex, axis_name: AxisName | None):
    if 'on_add_axis' in self._var_metadata:
      self._var_metadata['on_add_axis'](self, axis_index, axis_name)

  def remove_axis(self, axis_index: AxisIndex, axis_name: AxisName | None):
    if 'on_remove_axis' in self._var_metadata:
      self._var_metadata['on_remove_axis'](self, axis_index, axis_name)

GraphVariableState = VariableState[VariableState[tp.Any]]

def _variable_state_flatten(x: VariableState[tp.Any], *, with_keys: bool):
  metadata = tuple(x.get_metadata().items())
  if with_keys:
    node = (jtu.GetAttrKey('value'), x.value)
  else:
    node = x.value

  return (node,), (x.type, metadata)


def _variable_state_unflatten(
  static: tuple[type[Variable[A]], tuple[tuple[str, tp.Any], ...]],
  children: tuple[A],
) -> VariableState[A]:
  return VariableState(
    type=static[0],
    value=children[0],
    **dict(static[1]),
  )


jtu.register_pytree_with_keys(
  VariableState,
  partial(_variable_state_flatten, with_keys=True),  # type: ignore
  _variable_state_unflatten,  # type: ignore
  flatten_func=partial(_variable_state_flatten, with_keys=False),  # type: ignore
)


def with_metadata(
  initializer: F,
  set_value_hooks: tp.Union[SetValueHook[A], tp.Sequence[SetValueHook[A]]] = (),
  get_value_hooks: tp.Union[SetValueHook[A], tp.Sequence[SetValueHook[A]]] = (),
  create_value_hooks: tp.Union[
    CreateValueHook[A], tp.Sequence[CreateValueHook[A]]
  ] = (),
  add_axis_hooks: tp.Union[
    AddAxisHook[Variable[A]], tp.Sequence[AddAxisHook[Variable[A]]]
  ] = (),
  remove_axis_hooks: tp.Union[
    RemoveAxisHook[Variable[A]],
    tp.Sequence[RemoveAxisHook[Variable[A]]],
  ] = (),
  **metadata: tp.Any,
) -> F:
  if set_value_hooks:
    if callable(set_value_hooks):
      set_value_hooks = (set_value_hooks,)
    else:
      set_value_hooks = tuple(set_value_hooks)
  else:
    set_value_hooks = ()

  if get_value_hooks:
    if callable(get_value_hooks):
      get_value_hooks = (get_value_hooks,)
    else:
      get_value_hooks = tuple(get_value_hooks)
  else:
    get_value_hooks = ()

  if create_value_hooks:
    if callable(create_value_hooks):
      create_value_hooks = (create_value_hooks,)
    else:
      create_value_hooks = tuple(create_value_hooks)
  else:
    create_value_hooks = ()

  if add_axis_hooks:
    if callable(add_axis_hooks):
      add_axis_hooks = (add_axis_hooks,)
    else:
      add_axis_hooks = tuple(add_axis_hooks)
  else:
    add_axis_hooks = ()

  if remove_axis_hooks:
    if callable(remove_axis_hooks):
      remove_axis_hooks = (remove_axis_hooks,)
    else:
      remove_axis_hooks = tuple(remove_axis_hooks)
  else:
    remove_axis_hooks = ()

  @functools.wraps(initializer)
  def wrapper(*args):
    return VariableMetadata(
      initializer(*args),
      set_value_hooks=set_value_hooks,
      get_value_hooks=get_value_hooks,
      create_value_hooks=create_value_hooks,
      add_axis_hooks=add_axis_hooks,
      remove_axis_hooks=remove_axis_hooks,
      metadata=metadata,
    )

  return wrapper  # type: ignore


def split_flat_state(
    flat_state: tp.Iterable[tuple[PathParts, Variable | VariableState]],
    filters: tuple[filterlib.Filter, ...],
) -> tuple[list[tuple[PathParts, Variable | VariableState]], ...]:
  predicates = filterlib.filters_to_predicates(filters)
  # we have n + 1 states, where n is the number of predicates
  # the last state is for values that don't match any predicate
  flat_states: tuple[list[tuple[PathParts, Variable | VariableState]], ...] = (
    tuple([] for _ in predicates)
  )

  for path, value in flat_state:
    for i, predicate in enumerate(predicates):
      if predicate(path, value):
        flat_states[i].append((path, value))
        break
    else:
      raise ValueError(
          'Non-exhaustive filters, got a non-empty remainder: '
          f'{path} -> {value}.'
          '\nUse `...` to match all remaining elements.'
      )

  return flat_states


###################################################
### Variable type/class <-> string name mapping ###
###################################################
# Assumption: the mapping is 1-1 and unique.

VariableTypeCache: dict[str, tp.Type[Variable[tp.Any]]] = {}


def variable_type_from_name(
  name: str,
  /,
  *,
  base: type[Variable[tp.Any]] = Variable,
  allow_register: bool = False,
) -> tp.Type[Variable[tp.Any]]:
  """Given a Linen-style collection name, get or create its NNX Variable class."""
  if name not in VariableTypeCache:
    if not allow_register:
      raise ValueError(
        f'Name {name} is not registered in the registry. '
        'To register a new name, use register_variable_name() '
        'or set allow_register=True.'
      )
    VariableTypeCache[name] = type(name, (base,), {})
  return VariableTypeCache[name]


def variable_name_from_type(
  typ: tp.Type[Variable[tp.Any]], /, *, allow_register: bool = False
) -> str:
  """Given an NNX Variable type, get its Linen-style collection name.

  Should output the exact inversed result of `variable_type_from_name()`."""
  for name, t in VariableTypeCache.items():
    if typ == t:
      return name

  if not allow_register:
    raise ValueError(
      f'Type {typ} is not registered in the registry. '
      'To register a new type, use register_variable_name() '
      'or set allow_register=True.'
    )
  name = typ.__name__
  if name in VariableTypeCache:
    raise ValueError(
      'Name {name} is already registered in the registry as {VariableTypeCache[name]}. '
      'It cannot be linked with this type {typ}.'
    )
  register_variable_name(name, typ)
  return name


class _Missing:
  pass


_MISSING = _Missing()


@tp.overload
def register_variable_name(
    name: str,
    typ: type[Variable[tp.Any]],
    *,
    overwrite: bool = False,
) -> type[Variable[tp.Any]]:
  ...


@tp.overload
def register_variable_name(
    name: str,
    *,
    overwrite: bool = False,
) -> tp.Callable[[type[Variable[tp.Any]]], type[Variable[tp.Any]]]:
  ...


def register_variable_name(
    name: str,
    typ: type[Variable[A]] | _Missing = _MISSING,
    *,
    overwrite=False,
) -> type[Variable[A]] | tp.Callable[[type[Variable[A]]], type[Variable[A]]]:
  """Register a pair of Linen collection name and its NNX type."""
  if typ is _MISSING:
    return partial(register_variable_name, name, overwrite=overwrite)
  typ = tp.cast(type[Variable[A]], typ)
  if not overwrite and name in VariableTypeCache:
    raise ValueError(f'Name {name} already mapped to type {VariableTypeCache[name]}. '
                     'To overwrite, call set_variable_name() with `overwrite=True`.')
  VariableTypeCache[name] = typ
  return typ


# add known variable type names
register_variable_name('params', Param)
register_variable_name('batch_stats', BatchStat)
register_variable_name('cache', Cache)
register_variable_name('intermediates', Intermediate)
register_variable_name('perturbations', Perturbation)
