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

import dataclasses
import enum
from typing import (
  Any,
  Generic,
  Protocol,
  TypeVar,
  runtime_checkable,
)
from collections.abc import Callable, Generator, Mapping

from flax.core import FrozenDict
from flax.errors import CursorFindError, TraverseTreeError

A = TypeVar('A')
Key = Any


@runtime_checkable
class Indexable(Protocol):
  def __getitem__(self, key) -> Any:
    ...


class AccessType(enum.Enum):
  ITEM = enum.auto()
  ATTR = enum.auto()


@dataclasses.dataclass
class ParentKey(Generic[A]):
  parent: 'Cursor[A]'
  key: Key
  access_type: AccessType


def is_named_tuple(obj):
  return (
    isinstance(obj, tuple)
    and hasattr(obj, '_fields')
    and hasattr(obj, '_asdict')
    and hasattr(obj, '_replace')
  )


def _traverse_tree(path, obj, *, update_fn=None, cond_fn=None):
  """Helper function for ``Cursor.apply_update`` and ``Cursor.find_all``.
  Exactly one of ``update_fn`` and ``cond_fn`` must be not None.

  - If ``update_fn`` is not None, then ``Cursor.apply_update`` is calling
    this function and ``_traverse_tree`` will return a generator where
    each generated element is of type Tuple[Tuple[Union[str, int], AccessType], Any].
    The first element is a tuple of the key path and access type where the
    change was applied from the ``update_fn``, and the second element is
    the newly modified value. If the generator is non-empty, then the
    tuple key path will always be non-empty as well.
  - If ``cond_fn`` is not None, then ``Cursor.find_all`` is calling this
    function and ``_traverse_tree`` will return a generator where each
    generated element is of type Tuple[Union[str, int], AccessType]. The
    tuple contains the key path and access type where the object was found
    that fulfilled the conditions of the ``cond_fn``.
  """
  if not (bool(update_fn) ^ bool(cond_fn)):
    raise TraverseTreeError(update_fn, cond_fn)

  if path:
    str_path = '/'.join(str(key) for key, _ in path)
    if update_fn:
      new_obj = update_fn(str_path, obj)
      if new_obj is not obj:
        yield path, new_obj
        return
    elif cond_fn(str_path, obj):  # type: ignore
      yield path
      return

  if isinstance(obj, (FrozenDict, dict)):
    items = obj.items()
    access_type = AccessType.ITEM
  elif is_named_tuple(obj):
    items = ((name, getattr(obj, name)) for name in obj._fields)  # type: ignore
    access_type = AccessType.ATTR
  elif isinstance(obj, (list, tuple)):
    items = enumerate(obj)
    access_type = AccessType.ITEM
  elif dataclasses.is_dataclass(obj):
    items = (
      (f.name, getattr(obj, f.name)) for f in dataclasses.fields(obj) if f.init
    )
    access_type = AccessType.ATTR
  else:
    return

  if update_fn:
    for key, value in items:
      yield from _traverse_tree(
        path + ((key, access_type),), value, update_fn=update_fn
      )
  else:
    for key, value in items:
      yield from _traverse_tree(
        path + ((key, access_type),), value, cond_fn=cond_fn
      )


class Cursor(Generic[A]):
  _obj: A
  _parent_key: ParentKey[A] | None
  _changes: dict[Any, 'Cursor[A]']

  def __init__(self, obj: A, parent_key: ParentKey[A] | None):
    # NOTE: we use `vars` here to avoid calling `__setattr__`
    # vars(self) = self.__dict__
    vars(self)['_obj'] = obj
    vars(self)['_parent_key'] = parent_key
    vars(self)['_changes'] = {}

  @property
  def _root(self) -> 'Cursor[A]':
    if self._parent_key is None:
      return self
    else:
      return self._parent_key.parent._root  # type: ignore

  @property
  def _path(self) -> str:
    if self._parent_key is None:
      return ''
    if self._parent_key.access_type == AccessType.ITEM:  # type: ignore
      if isinstance(self._parent_key.key, str):  # type: ignore
        key = "'" + self._parent_key.key + "'"  # type: ignore
      else:
        key = str(self._parent_key.key)  # type: ignore
      return self._parent_key.parent._path + '[' + key + ']'  # type: ignore
    # self.parent_key.access_type == AccessType.ATTR:
    return self._parent_key.parent._path + '.' + self._parent_key.key  # type: ignore

  def __getitem__(self, key) -> 'Cursor[A]':
    if key in self._changes:
      return self._changes[key]

    if not isinstance(self._obj, Indexable):
      raise TypeError(f'Cannot index into {self._obj}')

    if isinstance(self._obj, Mapping) and key not in self._obj:
      raise KeyError(f'Key {key} not found in {self._obj}')

    if is_named_tuple(self._obj):
      return getattr(self, self._obj._fields[key])  # type: ignore

    child = Cursor(self._obj[key], ParentKey(self, key, AccessType.ITEM))
    self._changes[key] = child
    return child

  def __getattr__(self, name) -> 'Cursor[A]':
    if name in self._changes:
      return self._changes[name]

    if not hasattr(self._obj, name):
      raise AttributeError(f'Attribute {name} not found in {self._obj}')

    child = Cursor(
      getattr(self._obj, name), ParentKey(self, name, AccessType.ATTR)
    )
    self._changes[name] = child
    return child

  def __setitem__(self, key, value):
    if is_named_tuple(self._obj):
      return setattr(self, self._obj._fields[key], value)  # type: ignore
    self._changes[key] = Cursor(value, ParentKey(self, key, AccessType.ITEM))

  def __setattr__(self, name, value):
    self._changes[name] = Cursor(value, ParentKey(self, name, AccessType.ATTR))

  def set(self, value) -> A:
    """Set a new value for an attribute, property, element or entry
    in the Cursor object and return a copy of the original object,
    containing the new set value.

    Example::

      >>> from flax.cursor import cursor
      >>> from flax.training import train_state
      >>> import optax

      >>> dict_obj = {'a': 1, 'b': (2, 3), 'c': [4, 5]}
      >>> modified_dict_obj = cursor(dict_obj)['b'][0].set(10)
      >>> assert modified_dict_obj == {'a': 1, 'b': (10, 3), 'c': [4, 5]}

      >>> state = train_state.TrainState.create(
      ...     apply_fn=lambda x: x,
      ...     params=dict_obj,
      ...     tx=optax.adam(1e-3),
      ... )
      >>> modified_state = cursor(state).params['b'][1].set(10)
      >>> assert modified_state.params == {'a': 1, 'b': (2, 10), 'c': [4, 5]}

    Args:
      value: the value used to set an attribute, property, element or entry in the Cursor object
    Returns:
      A copy of the original object with the new set value.
    """
    if self._parent_key is None:
      return value
    parent, key = self._parent_key.parent, self._parent_key.key  # type: ignore
    parent._changes[key] = Cursor(value, self._parent_key)
    return parent._root.build()

  def build(self) -> A:
    """Create and return a copy of the original object with accumulated changes.
    This method is to be called after making changes to the Cursor object.

    .. note::
      The new object is built bottom-up, the changes will be first applied
      to the leaf nodes, and then its parent, all the way up to the root.

    Example::

      >>> from flax.cursor import cursor
      >>> from flax.training import train_state
      >>> import optax

      >>> dict_obj = {'a': 1, 'b': (2, 3), 'c': [4, 5]}
      >>> c = cursor(dict_obj)
      >>> c['b'][0] = 10
      >>> c['a'] = (100, 200)
      >>> modified_dict_obj = c.build()
      >>> assert modified_dict_obj == {'a': (100, 200), 'b': (10, 3), 'c': [4, 5]}

      >>> state = train_state.TrainState.create(
      ...     apply_fn=lambda x: x,
      ...     params=dict_obj,
      ...     tx=optax.adam(1e-3),
      ... )
      >>> new_fn = lambda x: x + 1
      >>> c = cursor(state)
      >>> c.params['b'][1] = 10
      >>> c.apply_fn = new_fn
      >>> modified_state = c.build()
      >>> assert modified_state.params == {'a': 1, 'b': (2, 10), 'c': [4, 5]}
      >>> assert modified_state.apply_fn == new_fn

    Returns:
      A copy of the original object with the accumulated changes.
    """
    changes = {
      key: child.build() if isinstance(child, Cursor) else child
      for key, child in self._changes.items()
    }
    if isinstance(self._obj, FrozenDict):
      obj = self._obj.copy(changes)  # type: ignore
    elif isinstance(self._obj, (dict, list)):
      obj = self._obj.copy()  # type: ignore
      for key, value in changes.items():
        obj[key] = value
    elif is_named_tuple(self._obj):
      obj = self._obj._replace(**changes)  # type: ignore
    elif isinstance(self._obj, tuple):
      obj = list(self._obj)  # type: ignore
      for key, value in changes.items():
        obj[key] = value
      obj = tuple(obj)  # type: ignore
    elif dataclasses.is_dataclass(self._obj):
      obj = dataclasses.replace(self._obj, **changes)  # type: ignore
    else:
      obj = self._obj  # type: ignore
    return obj  # type: ignore

  def apply_update(
    self,
    update_fn: Callable[[str, Any], Any],
  ) -> 'Cursor[A]':
    """Traverse the Cursor object and record conditional changes recursively via an ``update_fn``.
    The changes are recorded in the Cursor object's ``._changes`` dictionary. To generate a copy
    of the original object with the accumulated changes, call the ``.build`` method after calling
    ``.apply_update``.

    The ``update_fn`` has a function signature of ``(str, Any) -> Any``:

    - The input arguments are the current key path (in the form of a string delimited
      by ``'/'``) and value at that current key path
    - The output is the new value (either modified by the ``update_fn`` or same as the
      input value if the condition wasn't fulfilled)

    .. note::
      - If the ``update_fn`` returns a modified value, this method will not recurse any further
        down that branch to record changes. For example, if we intend to replace an attribute that points
        to a dictionary with an int, we don't need to look for further changes inside the dictionary,
        since the dictionary will be replaced anyways.
      - The ``is`` operator is used to determine whether the return value is modified (by comparing it
        to the input value). Therefore if the ``update_fn`` modifies a mutable container (e.g. lists,
        dicts, etc.) and returns the same container, ``.apply_update`` will treat the returned value as
        unmodified as it contains the same ``id``. To avoid this, return a copy of the modified value.
      - ``.apply_update`` WILL NOT call the ``update_fn`` to the value at the top-most level of
        the pytree (i.e. the root node). The ``update_fn`` will first be called on the root node's
        children, and then the pytree traversal will continue recursively from there.

    Example::

      >>> import flax.linen as nn
      >>> from flax.cursor import cursor
      >>> import jax, jax.numpy as jnp

      >>> class Model(nn.Module):
      ...   @nn.compact
      ...   def __call__(self, x):
      ...     x = nn.Dense(3)(x)
      ...     x = nn.relu(x)
      ...     x = nn.Dense(3)(x)
      ...     x = nn.relu(x)
      ...     x = nn.Dense(3)(x)
      ...     x = nn.relu(x)
      ...     return x

      >>> params = Model().init(jax.random.key(0), jnp.empty((1, 2)))['params']

      >>> def update_fn(path, value):
      ...   '''Multiply all dense kernel params by 2 and add 1.
      ...   Subtract the Dense_1 bias param by 1.'''
      ...   if 'kernel' in path:
      ...     return value * 2 + 1
      ...   elif 'Dense_1' in path and 'bias' in path:
      ...     return value - 1
      ...   return value

      >>> c = cursor(params)
      >>> new_params = c.apply_update(update_fn).build()
      >>> for layer in ('Dense_0', 'Dense_1', 'Dense_2'):
      ...   assert (new_params[layer]['kernel'] == 2 * params[layer]['kernel'] + 1).all()
      ...   if layer == 'Dense_1':
      ...     assert (new_params[layer]['bias'] == params[layer]['bias'] - 1).all()
      ...   else:
      ...     assert (new_params[layer]['bias'] == params[layer]['bias']).all()

      >>> assert jax.tree_util.tree_all(
      ...       jax.tree_util.tree_map(
      ...           lambda x, y: (x == y).all(),
      ...           params,
      ...           Model().init(jax.random.key(0), jnp.empty((1, 2)))[
      ...               'params'
      ...           ],
      ...       )
      ...   ) # make sure original params are unchanged

    Args:
      update_fn: the function that will conditionally record changes to the Cursor object
    Returns:
      The current Cursor object with the recorded conditional changes specified by the
      ``update_fn``. To generate a copy of the original object with the accumulated
      changes, call the ``.build`` method after calling ``.apply_update``.
    """
    for path, value in _traverse_tree((), self._obj, update_fn=update_fn):
      child = self
      for key, access_type in path[:-1]:
        if access_type is AccessType.ITEM:
          child = child[key]
        else:  # access_type is AccessType.ATTR
          child = getattr(child, key)
      key, access_type = path[-1]
      if access_type is AccessType.ITEM:
        child[key] = value
      else:  # access_type is AccessType.ATTR
        setattr(child, key, value)

    return self

  def find(self, cond_fn: Callable[[str, Any], bool]) -> 'Cursor[A]':
    """Traverse the Cursor object and return a child Cursor object that fulfill the
    conditions in the ``cond_fn``. The ``cond_fn`` has a function signature of ``(str, Any) -> bool``:

    - The input arguments are the current key path (in the form of a string delimited
      by ``'/'``) and value at that current key path
    - The output is a boolean, denoting whether to return the child Cursor object at this path

    Raises a :meth:`CursorFindError <flax.errors.CursorFindError>` if no object or more
    than one object is found that fulfills the condition of the ``cond_fn``. We raise an
    error because the user should always expect this method to return the only object whose
    corresponding key path and value fulfill the condition of the ``cond_fn``.

    .. note::
      - If the ``cond_fn`` evaluates to True at a particular key path, this method will not recurse
        any further down that branch; i.e. this method will find and return the "earliest" child node
        that fulfills the condition in ``cond_fn`` in a particular key path
      - ``.find`` WILL NOT search the the value at the top-most level of the pytree (i.e. the root
        node). The ``cond_fn`` will be evaluated recursively, starting at the root node's children.

    Example::

      >>> import flax.linen as nn
      >>> from flax.cursor import cursor
      >>> import jax, jax.numpy as jnp

      >>> class Model(nn.Module):
      ...   @nn.compact
      ...   def __call__(self, x):
      ...     x = nn.Dense(3)(x)
      ...     x = nn.relu(x)
      ...     x = nn.Dense(3)(x)
      ...     x = nn.relu(x)
      ...     x = nn.Dense(3)(x)
      ...     x = nn.relu(x)
      ...     return x

      >>> params = Model().init(jax.random.PRNGKey(0), jnp.empty((1, 2)))['params']

      >>> def cond_fn(path, value):
      ...   '''Find the second dense layer params.'''
      ...   return 'Dense_1' in path

      >>> new_params = cursor(params).find(cond_fn)['bias'].set(params['Dense_1']['bias'] + 1)

      >>> for layer in ('Dense_0', 'Dense_1', 'Dense_2'):
      ...   if layer == 'Dense_1':
      ...     assert (new_params[layer]['bias'] == params[layer]['bias'] + 1).all()
      ...   else:
      ...     assert (new_params[layer]['bias'] == params[layer]['bias']).all()

      >>> c = cursor(params)
      >>> c2 = c.find(cond_fn)
      >>> c2['kernel'] += 2
      >>> c2['bias'] += 2
      >>> new_params = c.build()

      >>> for layer in ('Dense_0', 'Dense_1', 'Dense_2'):
      ...   if layer == 'Dense_1':
      ...     assert (new_params[layer]['kernel'] == params[layer]['kernel'] + 2).all()
      ...     assert (new_params[layer]['bias'] == params[layer]['bias'] + 2).all()
      ...   else:
      ...     assert (new_params[layer]['kernel'] == params[layer]['kernel']).all()
      ...     assert (new_params[layer]['bias'] == params[layer]['bias']).all()

      >>> assert jax.tree_util.tree_all(
      ...       jax.tree_util.tree_map(
      ...           lambda x, y: (x == y).all(),
      ...           params,
      ...           Model().init(jax.random.PRNGKey(0), jnp.empty((1, 2)))[
      ...               'params'
      ...           ],
      ...       )
      ...   ) # make sure original params are unchanged

    Args:
      cond_fn: the function that will conditionally find child Cursor objects
    Returns:
      A child Cursor object that fulfills the condition in the ``cond_fn``.
    """
    generator = self.find_all(cond_fn)
    try:
      cursor = next(generator)
    except StopIteration:
      raise CursorFindError()
    try:
      cursor2 = next(generator)
      raise CursorFindError(cursor, cursor2)
    except StopIteration:
      return cursor

  def find_all(
    self, cond_fn: Callable[[str, Any], bool]
  ) -> Generator['Cursor[A]', None, None]:
    """Traverse the Cursor object and return a generator of child Cursor objects that fulfill the
    conditions in the ``cond_fn``. The ``cond_fn`` has a function signature of ``(str, Any) -> bool``:

    - The input arguments are the current key path (in the form of a string delimited
      by ``'/'``) and value at that current key path
    - The output is a boolean, denoting whether to return the child Cursor object at this path

    .. note::
      - If the ``cond_fn`` evaluates to True at a particular key path, this method will not recurse
        any further down that branch; i.e. this method will find and return the "earliest" child nodes
        that fulfill the condition in ``cond_fn`` in a particular key path
      - ``.find_all`` WILL NOT search the the value at the top-most level of the pytree (i.e. the root
        node). The ``cond_fn`` will be evaluated recursively, starting at the root node's children.

    Example::

      >>> import flax.linen as nn
      >>> from flax.cursor import cursor
      >>> import jax, jax.numpy as jnp

      >>> class Model(nn.Module):
      ...   @nn.compact
      ...   def __call__(self, x):
      ...     x = nn.Dense(3)(x)
      ...     x = nn.relu(x)
      ...     x = nn.Dense(3)(x)
      ...     x = nn.relu(x)
      ...     x = nn.Dense(3)(x)
      ...     x = nn.relu(x)
      ...     return x

      >>> params = Model().init(jax.random.PRNGKey(0), jnp.empty((1, 2)))['params']

      >>> def cond_fn(path, value):
      ...   '''Find all dense layer params.'''
      ...   return 'Dense' in path

      >>> c = cursor(params)
      >>> for dense_params in c.find_all(cond_fn):
      ...   dense_params['bias'] += 1
      >>> new_params = c.build()

      >>> for layer in ('Dense_0', 'Dense_1', 'Dense_2'):
      ...   assert (new_params[layer]['bias'] == params[layer]['bias'] + 1).all()

      >>> assert jax.tree_util.tree_all(
      ...       jax.tree_util.tree_map(
      ...           lambda x, y: (x == y).all(),
      ...           params,
      ...           Model().init(jax.random.PRNGKey(0), jnp.empty((1, 2)))[
      ...               'params'
      ...           ],
      ...       )
      ...   ) # make sure original params are unchanged

    Args:
      cond_fn: the function that will conditionally find child Cursor objects
    Returns:
      A generator of child Cursor objects that fulfill the condition in the ``cond_fn``.
    """
    for path in _traverse_tree((), self._obj, cond_fn=cond_fn):
      child = self
      for key, access_type in path:
        if access_type is AccessType.ITEM:
          child = child[key]
        else:  # access_type is AccessType.ATTR
          child = getattr(child, key)
      yield child

  def __str__(self):
    return str(self._obj)

  def __repr__(self):
    return self._pretty_repr()

  def _pretty_repr(self, indent=2, _prefix_indent=0):
    s = 'Cursor(\n'
    obj_str = repr(self._obj).replace(
      '\n', '\n' + ' ' * (_prefix_indent + indent)
    )
    s += ' ' * (_prefix_indent + indent) + f'_obj={obj_str},\n'
    s += ' ' * (_prefix_indent + indent) + '_changes={'
    if self._changes:
      s += '\n'
      for key in self._changes:
        str_key = repr(key)
        prefix = ' ' * (_prefix_indent + 2 * indent) + str_key + ': '
        s += (
          prefix
          + self._changes[key]._pretty_repr(
            indent=indent, _prefix_indent=len(prefix)
          )
          + ',\n'
        )
      s = s[
        :-2
      ]  # remove comma and newline character for last element in self._changes
      s += '\n' + ' ' * (_prefix_indent + indent) + '}\n'
    else:
      s += '}\n'
    s += ' ' * _prefix_indent + ')'
    return s

  def __len__(self):
    return len(self._obj)

  def __iter__(self):
    if isinstance(self._obj, (tuple, list)):
      return (self[i] for i in range(len(self._obj)))
    else:
      raise NotImplementedError(
        '__iter__ method only implemented for tuples and lists, not type'
        f' {type(self._obj)}'
      )

  def __reversed__(self):
    if isinstance(self._obj, (tuple, list)):
      return (self[i] for i in range(len(self._obj) - 1, -1, -1))
    else:
      raise NotImplementedError(
        '__reversed__ method only implemented for tuples and lists, not type'
        f' {type(self._obj)}'
      )

  def __add__(self, other):
    return self._obj + other

  def __sub__(self, other):
    return self._obj - other

  def __mul__(self, other):
    return self._obj * other

  def __matmul__(self, other):
    return self._obj @ other

  def __truediv__(self, other):
    return self._obj / other

  def __floordiv__(self, other):
    return self._obj // other

  def __mod__(self, other):
    return self._obj % other

  def __divmod__(self, other):
    return divmod(self._obj, other)

  def __pow__(self, other):
    return pow(self._obj, other)

  def __lshift__(self, other):
    return self._obj << other

  def __rshift__(self, other):
    return self._obj >> other

  def __and__(self, other):
    return self._obj & other

  def __xor__(self, other):
    return self._obj ^ other

  def __or__(self, other):
    return self._obj | other

  def __radd__(self, other):
    return other + self._obj

  def __rsub__(self, other):
    return other - self._obj

  def __rmul__(self, other):
    return other * self._obj

  def __rmatmul__(self, other):
    return other @ self._obj

  def __rtruediv__(self, other):
    return other / self._obj

  def __rfloordiv__(self, other):
    return other // self._obj

  def __rmod__(self, other):
    return other % self._obj

  def __rdivmod__(self, other):
    return divmod(other, self._obj)

  def __rpow__(self, other):
    return pow(other, self._obj)

  def __rlshift__(self, other):
    return other << self._obj

  def __rrshift__(self, other):
    return other >> self._obj

  def __rand__(self, other):
    return other & self._obj

  def __rxor__(self, other):
    return other ^ self._obj

  def __ror__(self, other):
    return other | self._obj

  def __neg__(self):
    return -self._obj

  def __pos__(self):
    return +self._obj

  def __abs__(self):
    return abs(self._obj)

  def __invert__(self):
    return ~self._obj

  def __round__(self, ndigits=None):
    return round(self._obj, ndigits)

  def __lt__(self, other):
    return self._obj < other

  def __le__(self, other):
    return self._obj <= other

  def __eq__(self, other):
    return self._obj == other

  def __ne__(self, other):
    return self._obj != other

  def __gt__(self, other):
    return self._obj > other

  def __ge__(self, other):
    return self._obj >= other


def cursor(obj: A) -> Cursor[A]:
  """Wrap :class:`Cursor <flax.cursor.Cursor>` over ``obj`` and return it.
  Changes can then be applied to the Cursor object in the following ways:

  - single-line change via the ``.set`` method
  - multiple changes, and then calling the ``.build`` method
  - multiple changes conditioned on the pytree path and node value via the
    ``.apply_update`` method, and then calling the ``.build`` method

  ``.set`` example::

    >>> from flax.cursor import cursor

    >>> dict_obj = {'a': 1, 'b': (2, 3), 'c': [4, 5]}
    >>> modified_dict_obj = cursor(dict_obj)['b'][0].set(10)
    >>> assert modified_dict_obj == {'a': 1, 'b': (10, 3), 'c': [4, 5]}

  ``.build`` example::

    >>> from flax.cursor import cursor

    >>> dict_obj = {'a': 1, 'b': (2, 3), 'c': [4, 5]}
    >>> c = cursor(dict_obj)
    >>> c['b'][0] = 10
    >>> c['a'] = (100, 200)
    >>> modified_dict_obj = c.build()
    >>> assert modified_dict_obj == {'a': (100, 200), 'b': (10, 3), 'c': [4, 5]}

  ``.apply_update`` example::

    >>> from flax.cursor import cursor
    >>> from flax.training import train_state
    >>> import optax

    >>> def update_fn(path, value):
    ...   '''Replace params with empty dictionary.'''
    ...   if 'params' in path:
    ...     return {}
    ...   return value

    >>> state = train_state.TrainState.create(
    ...     apply_fn=lambda x: x,
    ...     params={'a': 1, 'b': 2},
    ...     tx=optax.adam(1e-3),
    ... )
    >>> c = cursor(state)
    >>> state2 = c.apply_update(update_fn).build()
    >>> assert state2.params == {}
    >>> assert state.params == {'a': 1, 'b': 2} # make sure original params are unchanged

  If the underlying ``obj`` is a ``list`` or ``tuple``, iterating over the Cursor object
  to get the child Cursors is also possible::

    >>> from flax.cursor import cursor

    >>> c = cursor(((1, 2), (3, 4)))
    >>> for child_c in c:
    ...   child_c[1] *= -1
    >>> assert c.build() == ((1, -2), (3, -4))

  View the docstrings for each method to see more examples of their usage.

  Args:
    obj: the object you want to wrap the Cursor in
  Returns:
    A Cursor object wrapped around obj.
  """
  return Cursor(obj, None)
