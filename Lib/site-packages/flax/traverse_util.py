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

"""A utility for traversing immutable datastructures.

A Traversal can be used to iterate and update complex data structures.
Traversals take in an object and return a subset of its contents.
For example, a Traversal could select an attribute of an object::

  >>> from flax import traverse_util
  >>> import dataclasses

  >>> @dataclasses.dataclass
  ... class Foo:
  ...   foo: int = 0
  ...   bar: int = 0
  ...
  >>> x = Foo(foo=1)
  >>> iterator = traverse_util.TraverseAttr('foo').iterate(x)
  >>> list(iterator)
  [1]

More complex traversals can be constructed using composition.
It is often useful to start from the identity traversal and use a method chain
to construct the intended Traversal::

  >>> data = [{'foo': 1, 'bar': 2}, {'foo': 3, 'bar': 4}]
  >>> traversal = traverse_util.t_identity.each()['foo']
  >>> iterator = traversal.iterate(data)
  >>> list(iterator)
  [1, 3]

Traversals can also be used to make changes using the ``update`` method::

  >>> data = {'foo': Foo(bar=2)}
  >>> traversal = traverse_util.t_identity['foo'].bar
  >>> data = traversal.update(lambda x: x + x, data)
  >>> data
  {'foo': Foo(foo=0, bar=4)}

Traversals never mutate the original data. Therefore, an update essentially
returns a copy of the data including the provided updates.
"""

import abc
import copy
import dataclasses
import warnings
from typing import Any
from collections.abc import Callable

import jax

import flax
from flax.core.scope import VariableDict
from flax.typing import PathParts

from . import struct


# the empty node is a struct.dataclass to be compatible with JAX.
@struct.dataclass
class _EmptyNode:
  pass


empty_node = _EmptyNode()


def flatten_dict(xs, keep_empty_nodes=False, is_leaf=None, sep=None):
  """Flatten a nested dictionary.

  The nested keys are flattened to a tuple.
  See ``unflatten_dict`` on how to restore the
  nested dictionary structure.

  Example::

    >>> from flax.traverse_util import flatten_dict

    >>> xs = {'foo': 1, 'bar': {'a': 2, 'b': {}}}
    >>> flat_xs = flatten_dict(xs)
    >>> flat_xs
    {('foo',): 1, ('bar', 'a'): 2}

  Note that empty dictionaries are ignored and
  will not be restored by ``unflatten_dict``.

  Args:
    xs: a nested dictionary
    keep_empty_nodes: replaces empty dictionaries
      with ``traverse_util.empty_node``.
    is_leaf: an optional function that takes the
      next nested dictionary and nested keys and
      returns True if the nested dictionary is a
      leaf (i.e., should not be flattened further).
    sep: if specified, then the keys of the returned
      dictionary will be ``sep``-joined strings (if
      ``None``, then keys will be tuples).
  Returns:
    The flattened dictionary.
  """
  assert isinstance(
    xs, (flax.core.FrozenDict, dict)
  ), f'expected (frozen)dict; got {type(xs)}'

  def _key(path):
    if sep is None:
      return path
    return sep.join(path)

  def _flatten(xs, prefix):
    if not isinstance(xs, (flax.core.FrozenDict, dict)) or (
      is_leaf and is_leaf(prefix, xs)
    ):
      return {_key(prefix): xs}
    result = {}
    is_empty = True
    for key, value in xs.items():
      is_empty = False
      path = prefix + (key,)
      result.update(_flatten(value, path))
    if keep_empty_nodes and is_empty:
      if prefix == ():  # when the whole input is empty
        return {}
      return {_key(prefix): empty_node}
    return result

  return _flatten(xs, ())


def unflatten_dict(xs, sep=None):
  """Unflatten a dictionary.

  See ``flatten_dict``

  Example::

    >>> flat_xs = {
    ...   ('foo',): 1,
    ...   ('bar', 'a'): 2,
    ... }
    >>> xs = unflatten_dict(flat_xs)
    >>> xs
    {'foo': 1, 'bar': {'a': 2}}

  Args:
    xs: a flattened dictionary
    sep: separator (same as used with ``flatten_dict()``).
  Returns:
    The nested dictionary.
  """
  assert isinstance(xs, dict), f'input is not a dict; it is a {type(xs)}'
  result = {}
  for path, value in xs.items():
    if sep is not None:
      path = path.split(sep)
    if value is empty_node:
      value = {}
    cursor = result
    for key in path[:-1]:
      if key not in cursor:
        cursor[key] = {}
      cursor = cursor[key]
    cursor[path[-1]] = value
  return result


def path_aware_map(
  f: Callable[[PathParts, Any], Any], nested_dict: VariableDict
) -> VariableDict:
  """A map function that operates over nested dictionary structures while taking
  the path to each leaf into account.

  Example::

    >>> import jax.numpy as jnp
    >>> from flax import traverse_util

    >>> params = {'a': {'x': 10, 'y': 3}, 'b': {'x': 20}}
    >>> f = lambda path, x: x + 5 if 'x' in path else -x
    >>> traverse_util.path_aware_map(f, params)
    {'a': {'x': 15, 'y': -3}, 'b': {'x': 25}}

  Args:
    f: A callable that takes in ``(path, value)`` arguments and maps them
      to a new value. Here ``path`` is a tuple of strings.
    nested_dict: A nested dictionary structure.

  Returns:
    A new nested dictionary structure with the mapped values.
  """
  flat = flatten_dict(nested_dict, keep_empty_nodes=True)
  return unflatten_dict(
    {k: f(k, v) if v is not empty_node else v for k, v in flat.items()}
  )


class Traversal(abc.ABC):
  """Base class for all traversals."""

  def __new__(cls, *args, **kwargs):
    # Must override __new__ instead of __init__ since this is an ABC
    warnings.warn(
      '`flax.traverse_util.Traversal` will be deprecated. If you are using '
      'it for `flax.optim`, use `optax` instead. Refer to the update guide '
      'https://flax.readthedocs.io/en/latest/guides/converting_and_upgrading/optax_update_guide.html '
      'for detailed instructions.',
      DeprecationWarning,
    )
    return super().__new__(cls)

  @abc.abstractmethod
  def update(self, fn, inputs):
    """Update the focused items.

    Args:
      fn: the callback function that maps each traversed item
        to its updated value.
      inputs: the object that should be traversed.
    Returns:
      A new object with the updated values.
    """
    pass

  @abc.abstractmethod
  def iterate(self, inputs):
    """Iterate over the values selected by this ``Traversal``.

    Args:
      inputs: the object that should be traversed.
    Returns:
      An iterator over the traversed values.
    """
    pass

  def set(self, values, inputs):
    """Overrides the values selected by the ``Traversal``.

    Args:
      values: a list containing the new values.
      inputs: the object that should be traversed.
    Returns:
      A new object with the updated values.
    """

    def update_fn(_):
      if not values:
        raise ValueError('Not enough values provided')
      return values.pop(0)

    y = self.update(update_fn, inputs)
    if values:
      raise ValueError('Too many values provided')
    return y

  def compose(self, other):
    """Compose two traversals."""
    return TraverseCompose(self, other)

  def merge(self, *traversals):
    """Compose an arbitrary number of traversals and merge the results."""
    return self.compose(TraverseMerge(*traversals))

  def each(self):
    """Traverse each item in the selected containers."""
    return self.compose(TraverseEach())

  def tree(self):
    """Traverse each item in a pytree."""
    return self.compose(TraverseTree())

  def filter(self, fn):
    """Filter the selected values."""
    return self.compose(TraverseFilter(fn))

  def __getattr__(self, attr):
    return self.compose(TraverseAttr(attr))

  def __getitem__(self, key):
    return self.compose(TraverseItem(key))


class TraverseId(Traversal):
  """The identity Traversal."""

  def update(self, fn, inputs):
    return fn(inputs)

  def iterate(self, inputs):
    yield inputs


with warnings.catch_warnings():
  warnings.simplefilter('ignore', DeprecationWarning)
  t_identity = TraverseId()


class TraverseMerge(Traversal):
  """Merges the selection from a set of traversals."""

  def __init__(self, *traversals):
    self._traversals = traversals

  def update(self, fn, inputs):
    for traversal in self._traversals:
      inputs = traversal.update(fn, inputs)
    return inputs

  def iterate(self, inputs):
    for traversal in self._traversals:
      yield from traversal.iterate(inputs)


class TraverseCompose(Traversal):
  """Compose two traversals."""

  def __init__(self, x, y):
    self._x = x
    self._y = y

  def update(self, fn, inputs):
    def update_fn(x):
      return self._y.update(fn, x)

    return self._x.update(update_fn, inputs)

  def iterate(self, inputs):
    for x in self._x.iterate(inputs):
      yield from self._y.iterate(x)


class TraverseFilter(Traversal):
  """Filter selected values based on a predicate."""

  def __init__(self, fn):
    self._fn = fn

  def update(self, fn, inputs):
    if self._fn(inputs):
      return fn(inputs)
    else:
      return inputs

  def iterate(self, inputs):
    if self._fn(inputs):
      yield inputs


def _is_namedtuple(t):
  return issubclass(t, tuple) and hasattr(t, '_fields')


class TraverseAttr(Traversal):
  """Traverse the attribute of an object."""

  def __init__(self, attr):
    self._attr = attr

  def update(self, fn, inputs):
    value = fn(getattr(inputs, self._attr))
    if _is_namedtuple(type(inputs)):
      return inputs._replace(**{self._attr: value})
    elif dataclasses.is_dataclass(inputs):
      return dataclasses.replace(inputs, **{self._attr: value})
    else:
      inputs = copy.copy(inputs)
      setattr(inputs, self._attr, value)
      return inputs

  def iterate(self, inputs):
    yield getattr(inputs, self._attr)


class TraverseItem(Traversal):
  """Traverse the item of an object."""

  def __init__(self, key):
    self._key = key

  def update(self, fn, inputs):
    if isinstance(inputs, tuple):
      ty = type(inputs)
      if isinstance(self._key, slice):
        sl = self._key
      else:
        sl = slice(self._key, self._key + 1)
      indices = set(range(*sl.indices(len(inputs))))

      args = [
        fn(inputs[i]) if i in indices else inputs[i] for i in range(len(inputs))
      ]
      if _is_namedtuple(ty):
        return ty(*args)
      else:
        return ty(args)
    else:
      xs = copy.copy(inputs)
      xs[self._key] = fn(xs[self._key])
      return xs

  def iterate(self, inputs):
    if isinstance(self._key, slice):
      yield from inputs[self._key]
    else:
      yield inputs[self._key]


class TraverseEach(Traversal):
  """Traverse each item of a container."""

  def update(self, fn, inputs):
    ty = type(inputs)
    if ty is dict:
      return {key: fn(val) for key, val in inputs.items()}
    if ty not in {list, tuple}:
      raise ValueError('Only the entries of a list or tuple can be traversed.')
    return ty(fn(x) for x in inputs)

  def iterate(self, inputs):
    if isinstance(inputs, dict):
      yield from inputs.values()
    else:
      yield from inputs


class TraverseTree(Traversal):
  """Traverse every item in a pytree."""

  def update(self, fn, inputs):
    return jax.tree_util.tree_map(fn, inputs)

  def iterate(self, inputs):
    yield from jax.tree_util.tree_leaves(inputs)


def _get_params_dict(inputs):
  if isinstance(inputs, (dict, flax.core.FrozenDict)):
    return flax.core.unfreeze(inputs)
  else:
    raise ValueError(
      'Can only traverse a flax Model instance or a nested dict, not '
      f'{type(inputs)}'
    )


def _sorted_items(x):
  """Returns items of a dict ordered by keys."""
  return sorted(x.items(), key=lambda x: x[0])


class ModelParamTraversal(Traversal):
  """Select model parameters using a name filter.

  This traversal operates on a nested dictionary of parameters and selects a
  subset based on the ``filter_fn`` argument.

  See :class:`flax.optim.MultiOptimizer` for an example of how to use
  :class:`ModelParamTraversal` to update subsets of the parameter tree with a
  specific optimizer.
  """

  def __init__(self, filter_fn):
    """Constructor a new ModelParamTraversal.

    Args:
      filter_fn: a function that takes a parameter's full name and its value and
        returns whether this parameter should be selected or not. The name of a
        parameter is determined by the module hierarchy and the parameter name
        (for example: '/module/sub_module/parameter_name').
    """
    self._filter_fn = filter_fn

  def iterate(self, inputs):
    params = _get_params_dict(inputs)
    flat_dict = flatten_dict(params)
    for key, value in _sorted_items(flat_dict):
      path = '/' + '/'.join(key)
      if self._filter_fn(path, value):
        yield value

  def update(self, fn, inputs):
    params = _get_params_dict(inputs)
    flat_dict = flatten_dict(params, keep_empty_nodes=True)
    new_dict = {}
    for key, value in _sorted_items(flat_dict):
      # empty_node is not an actual leave. It's just a stub for empty nodes
      # in the nested dict.
      if value is not empty_node:
        path = '/' + '/'.join(key)
        if self._filter_fn(path, value):
          value = fn(value)
      new_dict[key] = value
    new_params = unflatten_dict(new_dict)
    if isinstance(inputs, flax.core.FrozenDict):
      return flax.core.FrozenDict(new_params)
    else:
      return new_params
