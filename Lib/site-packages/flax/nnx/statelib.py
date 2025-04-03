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

import typing as tp
from collections.abc import MutableMapping
from functools import partial
import warnings

import jax
import jax.tree_util as jtu
import treescope  # type: ignore[import-not-found,import-untyped]

from flax.nnx import filterlib, reprlib, traversals, variablelib
from flax.typing import Key, PathParts

A = tp.TypeVar('A')
K = tp.TypeVar('K', bound=tp.Hashable)
S = tp.TypeVar('S', bound='State')
V = tp.TypeVar('V')

ExtractValueFn = tp.Callable[[tp.Any], tp.Any]
SetValueFn = tp.Callable[[V, tp.Any], V]


class NestedStateRepr(reprlib.Representable):
  def __init__(self, state: State):
    self.state = state

  def __nnx_repr__(self):
    yield reprlib.Object('', kv_sep=': ', start='{', end='}')

    for r in self.state.__nnx_repr__():
      if isinstance(r, reprlib.Object):
        continue
      yield r

  def __treescope_repr__(self, path, subtree_renderer):
    children = {}
    for k, v in self.state.items():
      if isinstance(v, State):
        v = NestedStateRepr(v)
      children[k] = v
    # Render as the dictionary itself at the same path.
    return subtree_renderer(children, path=path)

class FlatState(tp.Sequence[tuple[PathParts, V]], reprlib.Representable):
  __slots__ = ('_keys', '_values')

  _keys: tuple[PathParts, ...]
  _values: list[V]

  def __init__(self, items: tp.Iterable[tuple[PathParts, V]], /, *, sort: bool):
    keys, values = [], []
    if sort:
      items = sorted(items)
    for key, value in items:
      keys.append(key)
      values.append(value)
    self._keys = tuple(keys)
    self._values = values

  @staticmethod
  def from_sorted_keys_values(
    keys: tuple[PathParts, ...], values: list[V], /
  ) -> FlatState[V]:
    flat_state = object.__new__(FlatState)
    flat_state._keys = keys
    flat_state._values = values
    return flat_state

  @property
  def paths(self) -> tp.Tuple[PathParts, ...]:
    return self._keys

  @property
  def leaves(self) -> list[V]:
    return self._values

  def __nnx_repr__(self):
    yield reprlib.Object(type='FlatState', kv_sep='', start='([', end='])')

    for value in self:
      yield reprlib.Attr('', value)

  @tp.overload
  def __getitem__(self, index: int) -> tuple[PathParts, V]: ...
  @tp.overload
  def __getitem__(self, index: slice) -> FlatState[V]: ...
  def __getitem__(
    self, index: int | slice
  ) -> tuple[PathParts, V] | FlatState[V]:
    if isinstance(index, int):
      return self._keys[index], self._values[index]
    return FlatState(zip(self._keys[index], self._values[index]), sort=False)

  def __len__(self) -> int:
    return len(self._keys)

  def __iter__(self) -> tp.Iterator[tuple[PathParts, V]]:
    return iter(zip(self._keys, self._values))

  def to_nested_state(self) -> State[Key, V]:
    return from_flat_state(self)

  @tp.overload
  def split(self, first: filterlib.Filter, /) -> FlatState[V]: ...

  @tp.overload
  def split(
    self,
    first: filterlib.Filter,
    second: filterlib.Filter,
    /,
    *filters: filterlib.Filter,
  ) -> tuple[FlatState[V], ...]: ...

  @tp.overload
  def split(
    self, /, *filters: filterlib.Filter
  ) -> tp.Union[FlatState[V], tuple[FlatState[V], ...]]: ...

  def split(  # type: ignore[misc]
    self, first: filterlib.Filter, /, *filters: filterlib.Filter
  ) -> tp.Union[FlatState[V], tuple[FlatState[V], ...]]:
    filters = (first, *filters)
    *flat_states_, rest = _split_state(self, *filters)

    if rest:
      raise ValueError(
        'Non-exhaustive filters, got a non-empty remainder: '
        f'{rest}.\nUse `...` to match all remaining elements.'
      )

    flat_states: FlatState[V] | tuple[FlatState[V], ...]
    if len(flat_states_) == 1:
      flat_states = flat_states_[0]
    else:
      flat_states = tuple(flat_states_)
    return flat_states  # type: ignore

  @tp.overload
  def filter(self, first: filterlib.Filter, /) -> FlatState[V]: ...

  @tp.overload
  def filter(
    self,
    first: filterlib.Filter,
    second: filterlib.Filter,
    /,
    *filters: filterlib.Filter,
  ) -> tuple[FlatState[V], ...]: ...

  def filter(
    self,
    first: filterlib.Filter,
    /,
    *filters: filterlib.Filter,
  ) -> tp.Union[FlatState[V], tuple[FlatState[V], ...]]:
    *flat_states_, _rest = _split_state(self, first, *filters)

    assert len(flat_states_) == len(filters) + 1

    flat_states: FlatState[V] | tuple[FlatState[V], ...]
    if len(flat_states_) == 1:
      flat_states = flat_states_[0]
    else:
      flat_states = tuple(flat_states_)

    return flat_states  # type: ignore

  @staticmethod
  def merge(
    flat_state: tp.Iterable[tuple[PathParts, V]],
    /,
    *flat_states: tp.Iterable[tuple[PathParts, V]],
  ) -> FlatState[V]:
    if not flat_states:
      if isinstance(flat_state, FlatState):
        return flat_state
      return FlatState(flat_state, sort=True)
    flat_states = (flat_state, *flat_states)

    return FlatState(
      (elem for flat_state in flat_states for elem in flat_state), sort=True
    )


def _flat_state_pytree_flatten(x: FlatState[V]):
  return x._values, x._keys


def _flat_state_pytree_unflatten(
  keys: tuple[PathParts, ...], values: list[V]
) -> FlatState[V]:
  flat_state = object.__new__(FlatState)
  flat_state._keys = keys
  flat_state._values = values
  return flat_state


jax.tree_util.register_pytree_node(
  FlatState,
  _flat_state_pytree_flatten,
  _flat_state_pytree_unflatten,
)


class State(MutableMapping[K, V], reprlib.Representable):
  """A pytree-like structure that contains a ``Mapping`` from hashable and
  comparable keys to leaves. Leaves can be of any type but :class:`VariableState`
  and :class:`Variable` are the most common.
  """

  def __init__(
    self,
    mapping: tp.Union[
      tp.Mapping[K, tp.Mapping | V],
      tp.Iterator[tuple[K, tp.Mapping | V]],
    ],
    /,
    *,
    _copy: bool = True,
  ):
    if _copy:
      _mapping = dict(mapping)
    else:
      if not isinstance(mapping, dict):
        raise ValueError(
          'Expected a dictionary when `_copy=False`, '
          f'got {type(mapping)} instead.'
        )
      _mapping = mapping

    if tp.TYPE_CHECKING:
      self._mapping = _mapping
    else:
      super().__setattr__('_mapping', _mapping)

  @property
  def raw_mapping(self) -> tp.Mapping[K, tp.Mapping[K, tp.Any] | V]:
    return self._mapping  # type: ignore

  def __contains__(self, key) -> bool:
    return key in self._mapping

  def __getitem__(self, key: K) -> State | V:  # type: ignore
    value = self._mapping[key]
    if isinstance(value, tp.Mapping):
      return type(self)(value, _copy=False)
    return value

  def __getattr__(self, key: K) -> State | V:  # type: ignore[misc]
    if '_mapping' not in vars(self) or key not in self._mapping:
      raise AttributeError(f"No attribute '{key}' in State")
    return self[key]

  def __setitem__(self, key: K, value: State | V) -> None:
    if key == '__orig_class__':
      object.__setattr__(self, key, value)  # type: ignore
    elif isinstance(value, State):
      self._mapping[key] = value._mapping
    else:
      self._mapping[key] = value

  __setattr__ = __setitem__  # type: ignore

  def __delitem__(self, key: K) -> None:
    del self._mapping[key]

  def __iter__(self) -> tp.Iterator[K]:
    return iter(self._mapping)

  def __len__(self) -> int:
    return len(self._mapping)

  def __nnx_repr__(self):
    yield reprlib.Object(type(self), kv_sep=': ', start='({', end='})')

    for k, v in self.items():
      if isinstance(v, State):
        v = NestedStateRepr(v)
      yield reprlib.Attr(repr(k), v)

  def __treescope_repr__(self, path, subtree_renderer):
    children = {}
    for k, v in self.items():
      if isinstance(v, State):
        v = NestedStateRepr(v)
      children[k] = v
    return treescope.repr_lib.render_dictionary_wrapper(
      object_type=type(self),
      wrapped_dict=children,
      path=path,
      subtree_renderer=subtree_renderer,
    )

  def map(self, f: tp.Callable[[tuple, V], V]) -> State[K, V]:
    warnings.warn(
      '`flax.nnx.State` will be deprecated and be replaced by the built-in '
      'Python dict. Please use the equivalent `nnx.map_state` instead.',
      DeprecationWarning,
    )
    return map_state(f, self)

  def flat_state(self) -> FlatState[V]:
    warnings.warn(
      '`flax.nnx.State` will be deprecated and be replaced by the built-in '
      'Python dict. Please use the equivalent `nnx.to_flat_state` instead.',
      DeprecationWarning,
    )
    return to_flat_state(self)

  @classmethod
  def from_flat_path(
    cls,
    flat_state: tp.Mapping[PathParts, V] | tp.Iterable[tuple[PathParts, V]],
    /,
  ):
    warnings.warn(
      '`flax.nnx.State` will be deprecated and be replaced by the built-in '
      'Python dict. Please use the equivalent `nnx.from_flat_state` instead.',
      DeprecationWarning,
    )
    return from_flat_state(flat_state, cls=cls)

  def to_pure_dict(self,
                   extract_fn: ExtractValueFn | None = None
                   ) -> dict[str, tp.Any]:
    warnings.warn(
      '`flax.nnx.State` will be deprecated and be replaced by the built-in '
      'Python dict. Please use the equivalent `nnx.to_pure_dict` instead.',
      DeprecationWarning,
    )
    return to_pure_dict(self, extract_fn)

  def replace_by_pure_dict(self,
                           pure_dict: dict[str, tp.Any],
                           replace_fn: SetValueFn | None = None):
    warnings.warn(
      '`flax.nnx.State` will be deprecated and be replaced by the built-in '
      'Python dict. Please use the equivalent `nnx.replace_by_pure_dict` '
      'instead.',
      DeprecationWarning,
    )
    return replace_by_pure_dict(self, pure_dict, replace_fn)

  @tp.overload
  def split(self, first: filterlib.Filter, /) -> State[K, V]: ...

  @tp.overload
  def split(
    self,
    first: filterlib.Filter,
    second: filterlib.Filter,
    /,
    *filters: filterlib.Filter,
  ) -> tuple[State[K, V], ...]: ...

  @tp.overload
  def split(
    self, /, *filters: filterlib.Filter
  ) -> tp.Union[State[K, V], tuple[State[K, V], ...]]: ...

  def split(  # type: ignore[misc]
    self, first: filterlib.Filter, /, *filters: filterlib.Filter
  ) -> tp.Union[State[K, V], tuple[State[K, V], ...]]:
    warnings.warn(
      '`flax.nnx.State` will be deprecated and be replaced by the built-in '
      'Python dict. Please use the equivalent `nnx.split_state` instead.',
      DeprecationWarning,
    )
    return split_state(self, first, *filters)

  @tp.overload
  def filter(
    self,
    first: filterlib.Filter,
    /,
  ) -> State[K, V]: ...

  @tp.overload
  def filter(
    self,
    first: filterlib.Filter,
    second: filterlib.Filter,
    /,
    *filters: filterlib.Filter,
  ) -> tuple[State[K, V], ...]: ...

  def filter(
    self,
    first: filterlib.Filter,
    /,
    *filters: filterlib.Filter,
  ) -> tp.Union[State[K, V], tuple[State[K, V], ...]]:
    warnings.warn(
      '`flax.nnx.State` will be deprecated and be replaced by the built-in '
      'Python dict. Please use the equivalent `nnx.filter_state` instead.',
      DeprecationWarning,
    )
    return filter_state(self, first, *filters)

  @classmethod
  def merge(cls, state: tp.Mapping[K, V], /, *states: tp.Mapping[K, V]):
    warnings.warn(
      '`flax.nnx.State` will be deprecated and be replaced by the built-in '
      'Python dict. Please use the equivalent `nnx.merge_state` instead.',
      DeprecationWarning,
    )
    return merge_state(state, *states)

  def __or__(self, other: State[K, V]) -> State[K, V]:
    if not other:
      return self
    return merge_state(self, other)

  def __sub__(self, other: State[K, V]) -> State[K, V]:
    warnings.warn(
      '`flax.nnx.State` will be deprecated and be replaced by the built-in '
      'Python dict. Please use the equivalent `nnx.diff` instead.',
      DeprecationWarning,
    )
    return diff(self, other)

  def __init_subclass__(cls) -> None:
    super().__init_subclass__()

    jax.tree_util.register_pytree_with_keys(
      cls,
      _state_flatten_with_keys,
      partial(_state_unflatten, cls),  # type: ignore[arg-type]
    )


def _state_flatten_with_keys(x: State):
  items = sorted(x._mapping.items())
  children = tuple((jtu.DictKey(key), value) for key, value in items)
  return children, tuple(key for key, _ in items)


def _state_unflatten(
  cls: type[S],
  static: tuple[K, ...],
  leaves: tuple[V, ...] | tuple[dict[K, V]],
):
  return cls(zip(static, leaves))


jax.tree_util.register_pytree_with_keys(
  State,
  _state_flatten_with_keys,
  partial(_state_unflatten, State),  # type: ignore[arg-type]
)


def map_state(f: tp.Callable[[tuple, tp.Any], tp.Any], state: State) -> State:
  flat_state = to_flat_state(state)
  result = [
    (path, f(path, variable_state)) for path, variable_state in flat_state
  ]
  return from_flat_state(result)


def to_flat_state(state: State) -> FlatState:
  return FlatState(traversals.flatten_to_sequence(state._mapping), sort=True)


def from_flat_state(
  flat_state: tp.Mapping[PathParts, V] | tp.Iterable[tuple[PathParts, V]],
  *, cls = State,  # for compatibility with State subclasses
) -> State:
  if not isinstance(flat_state, tp.Mapping):
    flat_state = dict(flat_state)
  nested_state = traversals.unflatten_mapping(flat_state)
  return cls(nested_state)


def to_pure_dict(
  state, extract_fn: ExtractValueFn | None = None
) -> dict[str, tp.Any]:
  # Works for nnx.Variable and nnx.VariableState
  if extract_fn is None:
    extract_fn = lambda x: x.value if hasattr(x, 'value') else x
  flat_values = {k: extract_fn(x) for k, x in to_flat_state(state)}
  return traversals.unflatten_mapping(flat_values)


def replace_by_pure_dict(
  state, pure_dict: dict[str, tp.Any], replace_fn: SetValueFn | None = None
):
  def try_convert_int(x):
    try:
      return int(x)
    except ValueError:
      return x

  # Works for nnx.Variable and nnx.VariableState
  if replace_fn is None:
    replace_fn = lambda x, v: x.replace(v) if hasattr(x, 'replace') else v
  current_flat = dict(to_flat_state(state))
  for kp, v in traversals.flatten_mapping(pure_dict).items():
    kp = tuple(map(try_convert_int, kp))
    if kp not in current_flat:
      raise ValueError(f'key in pure_dict not available in state: {kp}')
    current_flat[kp] = replace_fn(current_flat[kp], v)
  state.update(traversals.unflatten_mapping(current_flat))


@tp.overload
def split_state(state: State, first: filterlib.Filter, /) -> State: ...


@tp.overload
def split_state(
  state: State,
  first: filterlib.Filter,
  second: filterlib.Filter,
  /,
  *filters: filterlib.Filter,
) -> tuple[State, ...]: ...


@tp.overload
def split_state(
  state: State, /, *filters: filterlib.Filter
) -> tp.Union[State, tuple[State, ...]]: ...


def split_state(  # type: ignore[misc]
  state: State, first: filterlib.Filter, /, *filters: filterlib.Filter
) -> tp.Union[State, tuple[State, ...]]:
  """Split a ``State`` into one or more ``State``'s. The
  user must pass at least one ``Filter`` (i.e. :class:`Variable`),
  and the filters must be exhaustive (i.e. they must cover all
  :class:`Variable` types in the ``State``).

  Example usage::

    >>> from flax import nnx

    >>> class Model(nnx.Module):
    ...   def __init__(self, rngs):
    ...     self.batchnorm = nnx.BatchNorm(2, rngs=rngs)
    ...     self.linear = nnx.Linear(2, 3, rngs=rngs)
    ...   def __call__(self, x):
    ...     return self.linear(self.batchnorm(x))

    >>> model = Model(rngs=nnx.Rngs(0))
    >>> state = nnx.state(model)
    >>> param, batch_stats = nnx.split_state(state, nnx.Param, nnx.BatchStat)

  Arguments:
    first: The first filter
    *filters: The optional, additional filters to group the state into mutually exclusive substates.
  Returns:
    One or more ``States`` equal to the number of filters passed.
  """
  filters = (first, *filters)
  flat_states = _split_state(to_flat_state(state), *filters)
  *states_, rest = (state.to_nested_state() for state in flat_states)

  if rest:
    raise ValueError(
      'Non-exhaustive filters, got a non-empty remainder: '
      f'{rest}.\nUse `...` to match all remaining elements.'
    )

  states: State | tuple[State, ...]
  if len(states_) == 1:
    states = states_[0]
  else:
    states = tuple(states_)
  return states  # type: ignore



@tp.overload
def filter_state(
  state: State,
  first: filterlib.Filter,
  /,
) -> State: ...


@tp.overload
def filter_state(
  state: State,
  first: filterlib.Filter,
  second: filterlib.Filter,
  /,
  *filters: filterlib.Filter,
) -> tuple[State, ...]: ...


def filter_state(
  state: State,
  first: filterlib.Filter,
  /,
  *filters: filterlib.Filter,
) -> tp.Union[State, tuple[State, ...]]:
  """Filter a ``State`` into one or more ``State``'s. The
  user must pass at least one ``Filter`` (i.e. :class:`Variable`).
  This method is similar to :meth:`split() <flax.nnx.State.state.split>`,
  except the filters can be non-exhaustive.

  Example usage::

    >>> from flax import nnx

    >>> class Model(nnx.Module):
    ...   def __init__(self, rngs):
    ...     self.batchnorm = nnx.BatchNorm(2, rngs=rngs)
    ...     self.linear = nnx.Linear(2, 3, rngs=rngs)
    ...   def __call__(self, x):
    ...     return self.linear(self.batchnorm(x))

    >>> model = Model(rngs=nnx.Rngs(0))
    >>> state = nnx.state(model)
    >>> param = nnx.filter_state(state, nnx.Param)
    >>> batch_stats = nnx.filter_state(state, nnx.BatchStat)
    >>> param, batch_stats = nnx.filter_state(state, nnx.Param, nnx.BatchStat)

  Arguments:
    first: The first filter
    *filters: The optional, additional filters to group the state into mutually exclusive substates.
  Returns:
    One or more ``States`` equal to the number of filters passed.
  """
  flat_states = _split_state(to_flat_state(state), first, *filters)
  *states_, _rest = (state.to_nested_state() for state in flat_states)

  assert len(states_) == len(filters) + 1

  states: State | tuple[State, ...]
  if len(states_) == 1:
    states = states_[0]
  else:
    states = tuple(states_)

  return states  # type: ignore


def merge_state(state: tp.Mapping, /, *states: tp.Mapping,
  cls = State  # for compatibility with State subclasses
  ) -> State:
  """The inverse of :meth:`split() <flax.nnx.State.state.split>`.

  ``merge`` takes one or more ``State``'s and creates
  a new ``State``.

  Example usage::

    >>> from flax import nnx
    >>> import jax.numpy as jnp

    >>> class Model(nnx.Module):
    ...   def __init__(self, rngs):
    ...     self.batchnorm = nnx.BatchNorm(2, rngs=rngs)
    ...     self.linear = nnx.Linear(2, 3, rngs=rngs)
    ...   def __call__(self, x):
    ...     return self.linear(self.batchnorm(x))

    >>> model = Model(rngs=nnx.Rngs(0))
    >>> params, batch_stats = nnx.state(model, nnx.Param, nnx.BatchStat)
    >>> params['linear']['bias'].value += 1

    >>> state = nnx.merge_state(params, batch_stats)
    >>> nnx.update(model, state)
    >>> assert (model.linear.bias.value == jnp.array([1, 1, 1])).all()

  Args:
    state: A ``State`` object.
    *states: Additional ``State`` objects.
  Returns:
    The merged ``State``.
  """
  if not states:
    if isinstance(state, cls):
      return state
    return cls(state)

  states = (state, *states)

  new_state: dict[PathParts, tp.Any] = {}

  for state in states:
    new_state.update(traversals.flatten_mapping(state))  # type: ignore[attribute-error] # pytype is wrong here

  return from_flat_state(new_state, cls=cls)


def diff(state: State, other: State) -> State:
  if not other:
    return state

  self_flat = to_flat_state(state)
  other_flat = to_flat_state(other)
  diff = {k: v for k, v in self_flat.items() if k not in other_flat}

  return from_flat_state(diff)


def _split_state(
  flat_state: FlatState[V],
  *filters: filterlib.Filter,
) -> tuple[FlatState[V], ...]:
  for i, filter_ in enumerate(filters):
    if filter_ in (..., True) and i != len(filters) - 1:
      remaining_filters = filters[i + 1 :]
      if not all(f in (..., True) for f in remaining_filters):
        raise ValueError(
          '`...` or `True` can only be used as the last filters, '
          f'got {filter_} it at index {i}.'
        )
  predicates = tuple(map(filterlib.to_predicate, filters))

  # we have n + 1 states, where n is the number of predicates
  # the last state is for values that don't match any predicate
  flat_states: tuple[list[tuple[PathParts, V]], ...] = tuple(
    [] for _ in range(len(predicates) + 1)
  )

  for path, value in flat_state:
    for i, predicate in enumerate(predicates):
      if predicate(path, value):
        flat_states[i].append((path, value))  # type: ignore[index] # mypy is wrong here?
        break
    else:
      # if we didn't break, set leaf to last state
      flat_states[-1].append((path, value))  # type: ignore[index] # mypy is wrong here?

  return tuple(FlatState(flat_state, sort=False) for flat_state in flat_states)


def create_path_filters(state: State):
  flat_state = to_flat_state(state)
  value_paths: dict[tp.Any, set[PathParts]] = {}
  for path, value in flat_state:
    if isinstance(value, (variablelib.Variable, variablelib.VariableState)):
      value = value.value
    value_paths.setdefault(value, set()).add(path)
  return {filterlib.PathIn(*value_paths[value]): value for value in value_paths}