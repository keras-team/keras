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
from __future__ import annotations

import collections
from collections.abc import Callable, Hashable, Iterable, Sequence
import dataclasses
import difflib
import functools
from functools import partial
import operator as op
import textwrap
from typing import Any, NamedTuple, TypeVar, overload

from jax._src import traceback_util
from jax._src.lib import pytree
from jax._src.util import safe_zip, set_module
from jax._src.util import unzip2


export = set_module('jax.tree_util')

traceback_util.register_exclusion(__file__)

T = TypeVar("T")
Typ = TypeVar("Typ", bound=type[Any])
H = TypeVar("H", bound=Hashable)

Leaf = Any
PyTreeDef = pytree.PyTreeDef

default_registry = pytree.default_registry()
# Set __module__ and __name__, which allow this registry to be pickled by
# reference.
default_registry.__module__ = __name__
default_registry.__name__ = "default_registry"

# A copy of the default registry, where None is a leaf.
none_leaf_registry = pytree.PyTreeRegistry(
    enable_none=False, enable_tuple=True, enable_namedtuple=True,
    enable_list=True, enable_dict=True)
none_leaf_registry.__module__ = __name__
none_leaf_registry.__name__ = "none_leaf_registry"

# A special, internal pytree registry that includes everything in
# `default_registry`, plus internal Python-defined types that we want
# to teach the fast dispatch path ("C++ dispatch") how to flatten and
# unflatten. A key example is PRNG key arrays, which are currently a
# Python-defined class (in `jax._src.prng`). These ought to be a leaf
# node everywhere in the system (e.g. in Jaxpr), but we want to unpack
# and repack them across the fast dispatch boundary. If we were to
# skip registering such types here, the fast dispatch path would not
# know how to handle them as arguments. It would instead always
# indicate a "cache miss" and dispatch on the slow path.
dispatch_registry = pytree.PyTreeRegistry(
    enable_none=True, enable_tuple=True, enable_namedtuple=True,
    enable_list=True, enable_dict=True)
dispatch_registry.__module__ = __name__
dispatch_registry.__name__ = "dispatch_registry"


@export
def tree_flatten(tree: Any,
                 is_leaf: Callable[[Any], bool] | None = None
                 ) -> tuple[list[Leaf], PyTreeDef]:
  """Alias of :func:`jax.tree.flatten`."""
  return default_registry.flatten(tree, is_leaf)


@export
def tree_unflatten(treedef: PyTreeDef, leaves: Iterable[Leaf]) -> Any:
  """Alias of :func:`jax.tree.unflatten`."""
  return treedef.unflatten(leaves)


@export
def tree_leaves(tree: Any,
                is_leaf: Callable[[Any], bool] | None = None
                ) -> list[Leaf]:
  """Alias of :func:`jax.tree.leaves`."""
  return default_registry.flatten(tree, is_leaf)[0]


@export
def tree_structure(tree: Any,
                   is_leaf: None | (Callable[[Any],
                                              bool]) = None) -> PyTreeDef:
  """Alias of :func:`jax.tree.structure`."""
  return default_registry.flatten(tree, is_leaf)[1]


@export
def treedef_tuple(treedefs: Iterable[PyTreeDef]) -> PyTreeDef:
  """Makes a tuple treedef from an iterable of child treedefs.

  Args:
    treedefs: iterable of PyTree structures

  Returns:
    a single treedef representing a tuple of the structures

  Examples:
    >>> import jax
    >>> x = [1, 2, 3]
    >>> y = {'a': 4, 'b': 5}
    >>> x_tree = jax.tree.structure(x)
    >>> y_tree = jax.tree.structure(y)
    >>> xy_tree = jax.tree_util.treedef_tuple([x_tree, y_tree])
    >>> xy_tree == jax.tree.structure((x, y))
    True

  See Also:
    - :func:`jax.tree_util.treedef_children`
  """
  return pytree.tuple(default_registry, list(treedefs))


@export
def treedef_children(treedef: PyTreeDef) -> list[PyTreeDef]:
  """Return a list of treedefs for immediate children

  Args:
    treedef: a single PyTreeDef

  Returns:
    a list of PyTreeDefs representing the children of treedef.

  Examples:
    >>> import jax
    >>> x = [(1, 2), 3, {'a': 4}]
    >>> treedef = jax.tree.structure(x)
    >>> jax.tree_util.treedef_children(treedef)
    [PyTreeDef((*, *)), PyTreeDef(*), PyTreeDef({'a': *})]
    >>> _ == [jax.tree.structure(vals) for vals in x]
    True

  See Also:
    - :func:`jax.tree_util.treedef_tuple`
  """
  return treedef.children()


@export
def treedef_is_leaf(treedef: PyTreeDef) -> bool:
  """Return True if the treedef represents a leaf.

  Args:
    treedef: tree to check

  Returns:
    True if treedef is a leaf (i.e. has a single node); False otherwise.

  Examples:
    >>> import jax
    >>> tree1 = jax.tree.structure(1)
    >>> jax.tree_util.treedef_is_leaf(tree1)
    True
    >>> tree2 = jax.tree.structure([1, 2])
    >>> jax.tree_util.treedef_is_leaf(tree2)
    False
  """
  return treedef.num_nodes == 1


# treedef_is_strict_leaf is not exported.
def treedef_is_strict_leaf(treedef: PyTreeDef) -> bool:
  return treedef.num_nodes == 1 and treedef.num_leaves == 1


@export
def all_leaves(iterable: Iterable[Any],
               is_leaf: Callable[[Any], bool] | None = None) -> bool:
  """Tests whether all elements in the given iterable are all leaves.

  This function is useful in advanced cases, for example if a library allows
  arbitrary map operations on a flat iterable of leaves it may want to check
  if the result is still a flat iterable of leaves.

  Args:
    iterable: Iterable of leaves.

  Returns:
    A boolean indicating if all elements in the input are leaves.

  Examples:
    >>> import jax
    >>> tree = {"a": [1, 2, 3]}
    >>> assert all_leaves(jax.tree_util.tree_leaves(tree))
    >>> assert not all_leaves([tree])
  """
  if is_leaf is None:
    return pytree.all_leaves(default_registry, iterable)
  else:
    lst = list(iterable)
    return lst == tree_leaves(lst, is_leaf)


_Children = TypeVar("_Children", bound=Iterable[Any])
_AuxData = TypeVar("_AuxData", bound=Hashable)
KeyEntry = TypeVar("KeyEntry", bound=Any)
KeyLeafPair = tuple[KeyEntry, Any]
KeyLeafPairs = Iterable[KeyLeafPair]
KeyPath = tuple[KeyEntry, ...]


@export
def register_pytree_node(
    nodetype: type[T],
    flatten_func: Callable[[T], tuple[_Children, _AuxData]],
    unflatten_func: Callable[[_AuxData, _Children], T],
    flatten_with_keys_func: (
        Callable[[T], tuple[KeyLeafPairs, _AuxData]] | None
    ) = None,
) -> None:
  """Extends the set of types that are considered internal nodes in pytrees.

  See :ref:`example usage <pytrees>`.

  Args:
    nodetype: a Python type to register as a pytree.
    flatten_func: a function to be used during flattening, taking a value of
      type ``nodetype`` and returning a pair, with (1) an iterable for the
      children to be flattened recursively, and (2) some hashable auxiliary data
      to be stored in the treedef and to be passed to the ``unflatten_func``.
    unflatten_func: a function taking two arguments: the auxiliary data that was
      returned by ``flatten_func`` and stored in the treedef, and the
      unflattened children. The function should return an instance of
      ``nodetype``.

  See also:
    - :func:`~jax.tree_util.register_static`: simpler API for registering a static pytree.
    - :func:`~jax.tree_util.register_dataclass`: simpler API for registering a dataclass.
    - :func:`~jax.tree_util.register_pytree_with_keys`
    - :func:`~jax.tree_util.register_pytree_node_class`
    - :func:`~jax.tree_util.register_pytree_with_keys_class`

  Examples:
    First we'll define a custom type:

    >>> class MyContainer:
    ...   def __init__(self, size):
    ...     self.x = jnp.zeros(size)
    ...     self.y = jnp.ones(size)
    ...     self.size = size

    If we try using this in a JIT-compiled function, we'll get an error because JAX
    does not yet know how to handle this type:

    >>> m = MyContainer(size=5)
    >>> def f(m):
    ...   return m.x + m.y + jnp.arange(m.size)
    >>> jax.jit(f)(m)  # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
      ...
    TypeError: Cannot interpret value of type <class 'jax.tree_util.MyContainer'> as an abstract array; it does not have a dtype attribute

    In order to make our object recognized by JAX, we must register it as
    a pytree:

    >>> def flatten_func(obj):
    ...   children = (obj.x, obj.y)  # children must contain arrays & pytrees
    ...   aux_data = (obj.size,)  # aux_data must contain static, hashable data.
    ...   return (children, aux_data)
    ...
    >>> def unflatten_func(aux_data, children):
    ...   # Here we avoid `__init__` because it has extra logic we don't require:
    ...   obj = object.__new__(MyContainer)
    ...   obj.x, obj.y = children
    ...   obj.size, = aux_data
    ...   return obj
    ...
    >>> jax.tree_util.register_pytree_node(MyContainer, flatten_func, unflatten_func)

    Now with this defined, we can use instances of this type in JIT-compiled functions.

    >>> jax.jit(f)(m)
    Array([1., 2., 3., 4., 5.], dtype=float32)
  """
  default_registry.register_node(  # type: ignore[call-arg]
      nodetype, flatten_func, unflatten_func, flatten_with_keys_func   # type: ignore[arg-type]
  )
  none_leaf_registry.register_node(  # type: ignore[call-arg]
      nodetype, flatten_func, unflatten_func, flatten_with_keys_func   # type: ignore[arg-type]
  )
  dispatch_registry.register_node(  # type: ignore[call-arg]
      nodetype, flatten_func, unflatten_func, flatten_with_keys_func   # type: ignore[arg-type]
  )
  _registry[nodetype] = _RegistryEntry(flatten_func, unflatten_func)


@export
def register_pytree_node_class(cls: Typ) -> Typ:
  """Extends the set of types that are considered internal nodes in pytrees.

  This function is a thin wrapper around ``register_pytree_node``, and provides
  a class-oriented interface.

  Args:
    cls: a type to register as a pytree

  Returns:
    The input class ``cls`` is returned unchanged after being added to JAX's pytree
    registry. This return value allows ``register_pytree_node_class`` to be used as
    a decorator.

  See also:
    - :func:`~jax.tree_util.register_static`: simpler API for registering a static pytree.
    - :func:`~jax.tree_util.register_dataclass`: simpler API for registering a dataclass.
    - :func:`~jax.tree_util.register_pytree_node`
    - :func:`~jax.tree_util.register_pytree_with_keys`
    - :func:`~jax.tree_util.register_pytree_with_keys_class`

  Examples:
    Here we'll define a custom container that will be compatible with :func:`jax.jit`
    and other JAX transformations:

    >>> import jax
    >>> @jax.tree_util.register_pytree_node_class
    ... class MyContainer:
    ...   def __init__(self, x, y):
    ...     self.x = x
    ...     self.y = y
    ...   def tree_flatten(self):
    ...     return ((self.x, self.y), None)
    ...   @classmethod
    ...   def tree_unflatten(cls, aux_data, children):
    ...     return cls(*children)
    ...
    >>> m = MyContainer(jnp.zeros(4), jnp.arange(4))
    >>> def f(m):
    ...   return m.x + 2 * m.y
    >>> jax.jit(f)(m)
    Array([0., 2., 4., 6.], dtype=float32)
  """
  register_pytree_node(cls, op.methodcaller("tree_flatten"), cls.tree_unflatten)
  return cls


@export
def tree_map(f: Callable[..., Any],
             tree: Any,
             *rest: Any,
             is_leaf: Callable[[Any], bool] | None = None) -> Any:
  """Alias of :func:`jax.tree.map`."""
  leaves, treedef = tree_flatten(tree, is_leaf)
  all_leaves = [leaves] + [treedef.flatten_up_to(r) for r in rest]
  return treedef.unflatten(f(*xs) for xs in zip(*all_leaves))


@export
def build_tree(treedef: PyTreeDef, xs: Any) -> Any:
  """Build a treedef from a nested iterable structure

  Args:
    treedef: the PyTreeDef structure to build.
    xs: nested iterables matching the arity as the treedef

  Returns:
    object with structure defined by treedef

  See Also:
    - :func:`jax.tree.unflatten`

  Examples:
    >>> import jax
    >>> tree = [(1, 2), {'a': 3, 'b': 4}]
    >>> treedef = jax.tree.structure(tree)

    Both ``build_tree`` and :func:`jax.tree_util.tree_unflatten` can reconstruct
    the tree from new values, but ``build_tree`` takes these values in terms of
    a nested rather than flat structure:

    >>> jax.tree_util.build_tree(treedef, [[10, 11], [12, 13]])
    [(10, 11), {'a': 12, 'b': 13}]
    >>> jax.tree_util.tree_unflatten(treedef, [10, 11, 12, 13])
    [(10, 11), {'a': 12, 'b': 13}]
  """
  return treedef.from_iterable_tree(xs)


@export
def tree_transpose(outer_treedef: PyTreeDef, inner_treedef: PyTreeDef | None,
                   pytree_to_transpose: Any) -> Any:
  """Alias of :func:`jax.tree.transpose`."""
  flat, treedef = tree_flatten(pytree_to_transpose)
  if inner_treedef is None:
    inner_treedef = tree_structure(outer_treedef.flatten_up_to(pytree_to_transpose)[0])
  inner_size = inner_treedef.num_leaves
  outer_size = outer_treedef.num_leaves
  if treedef.num_leaves != (inner_size * outer_size):
    expected_treedef = outer_treedef.compose(inner_treedef)
    raise TypeError(f"Mismatch\n{treedef}\n != \n{expected_treedef}")
  iter_flat = iter(flat)
  lol = [
      [next(iter_flat) for _ in range(inner_size)] for __ in range(outer_size)
  ]
  transposed_lol = zip(*lol)
  subtrees = map(partial(tree_unflatten, outer_treedef), transposed_lol)
  return tree_unflatten(inner_treedef, subtrees)


# TODO(mattjj): remove the Python-side registry when the C++-side registry is
# sufficiently queryable that we can express _replace_nones. That may mean once
# we have a flatten_one function.
_RegistryEntry = collections.namedtuple("_RegistryEntry", ["to_iter", "from_iter"])
_registry: dict[type[Any], _RegistryEntry] = {
    tuple: _RegistryEntry(lambda xs: (xs, None), lambda _, xs: tuple(xs)),
    list: _RegistryEntry(lambda xs: (xs, None), lambda _, xs: list(xs)),
    dict: _RegistryEntry(lambda xs: unzip2(sorted(xs.items()))[::-1],
                         lambda keys, xs: dict(zip(keys, xs))),
    type(None): _RegistryEntry(lambda z: ((), None), lambda _, xs: None),
}

def _replace_nones(sentinel, tree):
  """Replaces ``None`` in ``tree`` with ``sentinel``."""
  leaves, treedef = none_leaf_registry.flatten(tree)
  leaves = map(lambda x: sentinel if x is None else x, leaves)
  return treedef.unflatten(leaves)


no_initializer = object()


@overload
def tree_reduce(function: Callable[[T, Any], T],
                tree: Any,
                *,
                is_leaf: Callable[[Any], bool] | None = None) -> T:
    ...


@overload
def tree_reduce(function: Callable[[T, Any], T],
                tree: Any,
                initializer: T,
                is_leaf: Callable[[Any], bool] | None = None) -> T:
    ...


@export
def tree_reduce(function: Callable[[T, Any], T],
                tree: Any,
                initializer: Any = no_initializer,
                is_leaf: Callable[[Any], bool] | None = None) -> T:
  """Alias of :func:`jax.tree.reduce`."""
  if initializer is no_initializer:
    return functools.reduce(function, tree_leaves(tree, is_leaf=is_leaf))
  else:
    return functools.reduce(function, tree_leaves(tree, is_leaf=is_leaf), initializer)


@export
def tree_all(tree: Any, *, is_leaf: Callable[[Any], bool] | None = None) -> bool:
  """Alias of :func:`jax.tree.all`."""
  return all(tree_leaves(tree, is_leaf=is_leaf))


class _HashableCallableShim:
  """Object that delegates __call__, __hash__, and __eq__ to another object."""

  def __init__(self, fun):
    self.fun = fun

  def __call__(self, *args, **kw):
    return self.fun(*args, **kw)

  def __hash__(self):
    return hash(self.fun)

  def __eq__(self, other):
    if isinstance(other, _HashableCallableShim):
      return self.fun == other.fun
    return self.fun == other

  def __repr__(self):
    return f'_HashableCallableShim({self.fun!r})'


@export
class Partial(functools.partial):
  """A version of functools.partial that works in pytrees.

  Use it for partial function evaluation in a way that is compatible with JAX's
  transformations, e.g., ``Partial(func, *args, **kwargs)``.

  (You need to explicitly opt-in to this behavior because we didn't want to give
  functools.partial different semantics than normal function closures.)

  For example, here is a basic usage of ``Partial`` in a manner similar to
  ``functools.partial``:

  >>> import jax.numpy as jnp
  >>> add_one = Partial(jnp.add, 1)
  >>> add_one(2)
  Array(3, dtype=int32, weak_type=True)

  Pytree compatibility means that the resulting partial function can be passed
  as an argument within transformed JAX functions, which is not possible with a
  standard ``functools.partial`` function:

  >>> from jax import jit
  >>> @jit
  ... def call_func(f, *args):
  ...   return f(*args)
  ...
  >>> call_func(add_one, 2)
  Array(3, dtype=int32, weak_type=True)

  Passing zero arguments to ``Partial`` effectively wraps the original function,
  making it a valid argument in JAX transformed functions:

  >>> call_func(Partial(jnp.add), 1, 2)
  Array(3, dtype=int32, weak_type=True)

  Had we passed ``jnp.add`` to ``call_func`` directly, it would have resulted in
  a ``TypeError``.

  Note that if the result of ``Partial`` is used in the context where the
  value is traced, it results in all bound arguments being traced when passed
  to the partially-evaluated function:

  >>> print_zero = Partial(print, 0)
  >>> print_zero()
  0
  >>> call_func(print_zero)  # doctest:+ELLIPSIS
  Traced<ShapedArray(int32[], weak_type=True)>with<DynamicJaxprTrace...>
  """

  def __new__(klass, func, *args, **kw):
    # In Python 3.10+, if func is itself a functools.partial instance,
    # functools.partial.__new__ would merge the arguments of this Partial
    # instance with the arguments of the func. We box func in a class that does
    # not (yet) have a `func` attribute to defeat this optimization, since we
    # care exactly which arguments are considered part of the pytree.
    if isinstance(func, functools.partial):
      original_func = func
      func = _HashableCallableShim(original_func)
      out = super().__new__(klass, func, *args, **kw)
      func.func = original_func.func
      func.args = original_func.args
      func.keywords = original_func.keywords
      return out
    else:
      return super().__new__(klass, func, *args, **kw)


register_pytree_node(
    Partial,
    lambda partial_: ((partial_.args, partial_.keywords), partial_.func),
    lambda func, xs: Partial(func, *xs[0], **xs[1]),
)


# broadcast_prefix is not exported.
def broadcast_prefix(prefix_tree: Any, full_tree: Any,
                     is_leaf: Callable[[Any], bool] | None = None
                     ) -> list[Any]:
  # If prefix_tree is not a tree prefix of full_tree, this code can raise a
  # ValueError; use prefix_errors to find disagreements and raise more precise
  # error messages.
  result = []
  num_leaves = lambda t: tree_structure(t).num_leaves
  add_leaves = lambda x, subtree: result.extend([x] * num_leaves(subtree))
  tree_map(add_leaves, prefix_tree, full_tree, is_leaf=is_leaf)
  return result


# flatten_one_level is not exported.
def flatten_one_level(tree: Any) -> tuple[Iterable[Any], Hashable]:
  """Flatten the given pytree node by one level.

  Args:
    tree: A valid pytree node, either built-in or registered via
      :func:`register_pytree_node` or related functions.

  Returns:
    A pair of the pytrees flattened children and its hashable metadata.

  Raises:
    ValueError: If the given pytree is not a built-in or registered container
    via ``register_pytree_node`` or ``register_pytree_with_keys``.

  Examples:
    >>> import jax
    >>> from jax._src.tree_util import flatten_one_level
    >>> flattened, meta = flatten_one_level({'a': [1, 2], 'b': {'c': 3}})
    >>> flattened
    ([1, 2], {'c': 3})
    >>> meta
    ('a', 'b')
  """
  out = default_registry.flatten_one_level(tree)
  if out is None:
    raise ValueError(f"can't tree-flatten type: {type(tree)}")
  else:
    return out


# flatten_one_level_with_keys is not exported.
def flatten_one_level_with_keys(
    tree: Any,
) -> tuple[Iterable[KeyLeafPair], Hashable]:
  """Flatten the given pytree node by one level, with keys."""
  out = default_registry.flatten_one_level_with_keys(tree)  # type: ignore
  if out is None:
    raise ValueError(f"can't tree-flatten type: {type(tree)}")
  else:
    return out


# prefix_errors is not exported
def prefix_errors(prefix_tree: Any, full_tree: Any,
                  is_leaf: Callable[[Any], bool] | None = None,
                  ) -> list[Callable[[str], ValueError]]:
  return list(_prefix_error((), prefix_tree, full_tree, is_leaf))


# equality_errors is not exported
def equality_errors(
    tree1: Any, tree2: Any, is_leaf: Callable[[Any], bool] | None = None,
) -> Iterable[tuple[KeyPath, str, str, str]]:
  """Helper to describe structural differences between two pytrees.

  Args:
    tree1, tree2: pytrees known to have different structure.

  Usage:

    raise Exception(
        "Value 1 and value 2 must have the same pytree structure, but they have "
        "the following structural differences:\n" +
        ("\n".join(
           f"   - {keystr(path)} is a {thing1} in value 1 and a {thing2} in "
           f" value 2, so {explanation}.\n"
           for path, thing1, thing2, explanation
           in equality_errors(val1, val2))))
  """
  yield from _equality_errors((), tree1, tree2, is_leaf)

def equality_errors_pytreedef(
    tree1: PyTreeDef,
    tree2: PyTreeDef) -> Iterable[tuple[KeyPath, str, str, str]]:
  """Like `equality_errors` but invoked on PyTreeDef."""
  # TODO(mattjj): make equality_errors not print type name, avoid metaclass
  leaf = type("LeafMeta", (type,), dict(__repr__=lambda _: "pytree leaf"))("Leaf", (), {})()
  return equality_errors(tree_unflatten(tree1, [leaf] * tree1.num_leaves),
                         tree_unflatten(tree2, [leaf] * tree2.num_leaves))

# TODO(mattjj): maybe share some logic with _prefix_error?
def _equality_errors(path, t1, t2, is_leaf):
  # If both are leaves, this isn't a structure equality error.
  if (treedef_is_strict_leaf(tree_structure(t1, is_leaf=is_leaf)) and
      treedef_is_strict_leaf(tree_structure(t2, is_leaf=is_leaf))): return

  # The trees may disagree because they are different types:
  if type(t1) != type(t2):
    yield path, str(type(t1)), str(type(t2)), 'their Python types differ'
    return  # no more errors to find

  # Or they may disagree because their roots have different numbers or keys of
  # children (with special-case handling of list/tuple):
  if isinstance(t1, (list, tuple)):
    assert type(t1) == type(t2)
    if len(t1) != len(t2):
      yield (path,
             f'{type(t1).__name__} of length {len(t1)}',
             f'{type(t2).__name__} of length {len(t2)}',
             'the lengths do not match')
      return  # no more errors to find
  t1_children, t1_meta = flatten_one_level(t1)
  t2_children, t2_meta = flatten_one_level(t2)
  t1_children = tuple(t1_children)
  t2_children = tuple(t2_children)
  t1_keys, t2_keys = _child_keys(t1), _child_keys(t2)
  try:
    diff = ' '.join(repr(k.key) for k in
                    set(t1_keys).symmetric_difference(set(t2_keys)))
  except:
    diff = ''
  if len(t1_children) != len(t2_children):
    yield (path,
           f'{type(t1)} with {len(t1_children)} child'
           f'{"ren" if len(t1_children) > 1 else ""}',
           f'{type(t2)} with {len(t2_children)} child'
           f'{"ren" if len(t2_children) > 1 else ""}',
           'the numbers of children do not match' +
           (diff and f', with the symmetric difference of key sets: {{{diff}}}')
           )
    return  # no more errors to find

  # Or they may disagree if their roots have different pytree metadata:
  if t1_meta != t2_meta:
    yield (path,
           f'{type(t1)} with pytree metadata {t1_meta}',
           f'{type(t2)} with pytree metadata {t2_meta}',
           'the pytree node metadata does not match')
    return  # no more errors to find

  # If the root types and numbers of children agree, there must be a mismatch in
  # a subtree, so recurse:
  assert t1_keys == t2_keys, \
      f"equal pytree nodes gave different tree keys: {t1_keys} and {t2_keys}"
  for k, c1, c2 in zip(t1_keys, t1_children, t2_children):
    yield from _equality_errors((*path, k), c1, c2, is_leaf)


SequenceKey: Any = pytree.SequenceKey  # type: ignore
DictKey: Any = pytree.DictKey  # type: ignore
GetAttrKey: Any = pytree.GetAttrKey  # type: ignore
FlattenedIndexKey: Any = pytree.FlattenedIndexKey  # type: ignore


@export
def keystr(keys: KeyPath):
  """Helper to pretty-print a tuple of keys.

  Args:
    keys: A tuple of ``KeyEntry`` or any class that can be converted to string.

  Returns:
    A string that joins all string representations of the keys.

  Examples:
    >>> import jax
    >>> keys = (0, 1, 'a', 'b')
    >>> jax.tree_util.keystr(keys)
    '01ab'
  """
  return ''.join(map(str, keys))


# TODO(ivyzheng): remove this after another jaxlib release.
class _RegistryWithKeypathsEntry(NamedTuple):
  flatten_with_keys: Callable[..., Any]
  unflatten_func: Callable[..., Any]


def _register_keypaths(
    ty: type[T], handler: Callable[[T], tuple[KeyEntry, ...]]
) -> None:
  def flatten_with_keys(xs):
    children, treedef = _registry[ty].to_iter(xs)
    return list(zip(handler(xs), children)), treedef
  if ty in _registry:
    _registry_with_keypaths[ty] = _RegistryWithKeypathsEntry(
        flatten_with_keys, _registry[ty].from_iter
    )

_registry_with_keypaths: dict[type[Any], _RegistryWithKeypathsEntry] = {}

_register_keypaths(
    tuple, lambda xs: tuple(SequenceKey(i) for i in range(len(xs)))
)
_register_keypaths(
    list, lambda xs: tuple(SequenceKey(i) for i in range(len(xs)))
)
_register_keypaths(dict, lambda xs: tuple(DictKey(k) for k in sorted(xs)))

_register_keypaths(
    collections.defaultdict, lambda x: tuple(DictKey(k) for k in x.keys())
)

_register_keypaths(
    collections.OrderedDict, lambda x: tuple(DictKey(k) for k in x.keys())
)


@export
def register_pytree_with_keys(
    nodetype: type[T],
    flatten_with_keys: Callable[[T], tuple[Iterable[KeyLeafPair], _AuxData]],
    unflatten_func: Callable[[_AuxData, Iterable[Any]], T],
    flatten_func: None | (Callable[[T], tuple[Iterable[Any], _AuxData]]) = None,
):
  """Extends the set of types that are considered internal nodes in pytrees.

  This is a more powerful alternative to ``register_pytree_node`` that allows
  you to access each pytree leaf's key path when flattening and tree-mapping.

  Args:
    nodetype: a Python type to treat as an internal pytree node.
    flatten_with_keys: a function to be used during flattening, taking a value
      of type ``nodetype`` and returning a pair, with (1) an iterable for tuples
      of each key path and its child, and (2) some hashable auxiliary data to be
      stored in the treedef and to be passed to the ``unflatten_func``.
    unflatten_func: a function taking two arguments: the auxiliary data that was
      returned by ``flatten_func`` and stored in the treedef, and the
      unflattened children. The function should return an instance of
      ``nodetype``.
    flatten_func: an optional function similar to ``flatten_with_keys``, but
      returns only children and auxiliary data. It must return the children
      in the same order as ``flatten_with_keys``, and return the same aux data.
      This argument is optional and only needed for faster traversal when
      calling functions without keys like ``tree_map`` and ``tree_flatten``.

  Examples:
    First we'll define a custom type:

    >>> class MyContainer:
    ...   def __init__(self, size):
    ...     self.x = jnp.zeros(size)
    ...     self.y = jnp.ones(size)
    ...     self.size = size

    Now register it using a key-aware flatten function:

    >>> from jax.tree_util import register_pytree_with_keys_class, GetAttrKey
    >>> def flatten_with_keys(obj):
    ...   children = [(GetAttrKey('x'), obj.x),
    ...               (GetAttrKey('y'), obj.y)]  # children must contain arrays & pytrees
    ...   aux_data = (obj.size,)  # aux_data must contain static, hashable data.
    ...   return children, aux_data
    ...
    >>> def unflatten(aux_data, children):
    ...   # Here we avoid `__init__` because it has extra logic we don't require:
    ...   obj = object.__new__(MyContainer)
    ...   obj.x, obj.y = children
    ...   obj.size, = aux_data
    ...   return obj
    ...
    >>> jax.tree_util.register_pytree_node(MyContainer, flatten_with_keys, unflatten)

    Now this can be used with functions like :func:`~jax.tree_util.tree_flatten_with_path`:

    >>> m = MyContainer(4)
    >>> leaves, treedef = jax.tree_util.tree_flatten_with_path(m)
  """
  if not flatten_func:
    def flatten_func_impl(tree):
      key_children, treedef = flatten_with_keys(tree)
      return [c for _, c in key_children], treedef
    flatten_func = flatten_func_impl

  register_pytree_node(
      nodetype, flatten_func, unflatten_func, flatten_with_keys
  )
  _registry_with_keypaths[nodetype] = _RegistryWithKeypathsEntry(
      flatten_with_keys, unflatten_func
  )


@export
def register_pytree_with_keys_class(cls: Typ) -> Typ:
  """Extends the set of types that are considered internal nodes in pytrees.

  This function is similar to ``register_pytree_node_class``, but requires a
  class that defines how it could be flattened with keys.

  It is a thin wrapper around ``register_pytree_with_keys``, and
  provides a class-oriented interface:

  Args:
    cls: a type to register as a pytree

  Returns:
    The input class ``cls`` is returned unchanged after being added to JAX's pytree
    registry. This return value allows ``register_pytree_node_class`` to be used as
    a decorator.

  See also:
    - :func:`~jax.tree_util.register_static`: simpler API for registering a static pytree.
    - :func:`~jax.tree_util.register_dataclass`: simpler API for registering a dataclass.
    - :func:`~jax.tree_util.register_pytree_node`
    - :func:`~jax.tree_util.register_pytree_with_keys`
    - :func:`~jax.tree_util.register_pytree_node_class`

  Examples:
    >>> from jax.tree_util import register_pytree_with_keys_class, GetAttrKey
    >>> @register_pytree_with_keys_class
    ... class Special:
    ...   def __init__(self, x, y):
    ...     self.x = x
    ...     self.y = y
    ...   def tree_flatten_with_keys(self):
    ...     return (((GetAttrKey('x'), self.x), (GetAttrKey('y'), self.y)), None)
    ...   @classmethod
    ...   def tree_unflatten(cls, aux_data, children):
    ...     return cls(*children)
  """
  flatten_func = (
      op.methodcaller("tree_flatten") if hasattr(cls, "tree_flatten") else None
  )
  register_pytree_with_keys(
      cls, op.methodcaller("tree_flatten_with_keys"), cls.tree_unflatten,
      flatten_func
  )
  return cls


@export
def register_dataclass(
    nodetype: Typ,
    data_fields: Sequence[str] | None = None,
    meta_fields: Sequence[str] | None = None,
    drop_fields: Sequence[str] = (),
) -> Typ:
  """Extends the set of types that are considered internal nodes in pytrees.

  This differs from ``register_pytree_with_keys_class`` in that the C++
  registries use the optimized C++ dataclass builtin instead of the argument
  functions.

  See :ref:`extending-pytrees` for more information about registering pytrees.

  Args:
    nodetype: a Python type to treat as an internal pytree node. This is assumed
      to have the semantics of a :obj:`~dataclasses.dataclass`: namely, class
      attributes represent the whole of the object state, and can be passed
      as keywords to the class constructor to create a copy of the object.
      All defined attributes should be listed among ``meta_fields`` or ``data_fields``.
    meta_fields: metadata field names: these are attributes which will be treated as
      {term}`static` when this pytree is passed to :func:`jax.jit`. ``meta_fields`` is
      optional only if ``nodetype`` is a dataclass, in which case individual fields can
      be marked static via :func:`dataclasses.field` (see examples below).
      Metadata fields *must* be static, hashable, immutable objects, as these objects
      are used to generate JIT cache keys. In particular, metadata fields cannot contain
      :class:`jax.Array` or :class:`numpy.ndarray` objects.
    data_fields: data field names: these are attributes which will be treated as non-static
      when this pytree is passed to :func:`jax.jit`. ``data_fields`` is optional only if
      ``nodetype`` is a dataclass, in which case fields are assumed data fields unless
      marked via :func:`dataclasses.field` (see examples below).
      Data fields *must* be JAX-compatible objects such as arrays (:class:`jax.Array`
      or :class:`numpy.ndarray`), scalars, or pytrees whose leaves are arrays or scalars.
      Note that ``None`` is a valid data field, as JAX recognizes this as an empty pytree.

  Returns:
    The input class ``nodetype`` is returned unchanged after being added to JAX's
    pytree registry, so that :func:`register_dataclass` can be used as a decorator.

  Examples:
    In JAX v0.4.35 or older, you must specify ``data_fields`` and ``meta_fields``
    in order to use this decorator:

    >>> import jax
    >>> from dataclasses import dataclass
    >>> from functools import partial
    ...
    >>> @partial(jax.tree_util.register_dataclass,
    ...          data_fields=['x', 'y'],
    ...          meta_fields=['op'])
    ... @dataclass
    ... class MyStruct:
    ...   x: jax.Array
    ...   y: jax.Array
    ...   op: str
    ...
    >>> m = MyStruct(x=jnp.ones(3), y=jnp.arange(3), op='add')
    >>> m
    MyStruct(x=Array([1., 1., 1.], dtype=float32), y=Array([0, 1, 2], dtype=int32), op='add')

    Starting in JAX v0.4.36, the ``data_fields`` and ``meta_fields`` arguments are optional
    for :func:`~dataclasses.dataclass` inputs, with fields defaulting to ``data_fields``
    unless marked as static using `static` metadata in :func:`dataclasses.field`.

    >>> import jax
    >>> from dataclasses import dataclass, field
    ...
    >>> @jax.tree_util.register_dataclass
    ... @dataclass
    ... class MyStruct:
    ...   x: jax.Array  # defaults to non-static data field
    ...   y: jax.Array  # defaults to non-static data field
    ...   op: str = field(metadata=dict(static=True))  # marked as static meta field.
    ...
    >>> m = MyStruct(x=jnp.ones(3), y=jnp.arange(3), op='add')
    >>> m
    MyStruct(x=Array([1., 1., 1.], dtype=float32), y=Array([0, 1, 2], dtype=int32), op='add')

    Once this class is registered, it can be used with functions in :mod:`jax.tree` and
    :mod:`jax.tree_util`:

    >>> leaves, treedef = jax.tree.flatten(m)
    >>> leaves
    [Array([1., 1., 1.], dtype=float32), Array([0, 1, 2], dtype=int32)]
    >>> treedef
    PyTreeDef(CustomNode(MyStruct[('add',)], [*, *]))
    >>> jax.tree.unflatten(treedef, leaves)
    MyStruct(x=Array([1., 1., 1.], dtype=float32), y=Array([0, 1, 2], dtype=int32), op='add')

    In particular, this registration allows ``m`` to be passed seamlessly through code
    wrapped in :func:`jax.jit` and other JAX transformations, with ``data_fields`` being
    treated as dynamic arguments, and ``meta_fields`` being treated as static arguments:

    >>> @jax.jit
    ... def compiled_func(m):
    ...   if m.op == 'add':
    ...     return m.x + m.y
    ...   else:
    ...     raise ValueError(f"{m.op=}")
    ...
    >>> compiled_func(m)
    Array([1., 2., 3.], dtype=float32)
  """
  if data_fields is None or meta_fields is None:
    if (data_fields is None) != (meta_fields is None):
      raise TypeError("register_dataclass: data_fields and meta_fields must both be specified"
                      f" when either is specified. Got {data_fields=} {meta_fields=}.")
    if not dataclasses.is_dataclass(nodetype):
      raise TypeError("register_dataclass: data_fields and meta_fields are required when"
                      f" nodetype is not a dataclass. Got {nodetype=}.")
    data_fields = [f.name for f in dataclasses.fields(nodetype)
                   if not f.metadata.get('static', False)]
    meta_fields = [f.name for f in dataclasses.fields(nodetype)
                   if f.metadata.get('static', False)]

  assert meta_fields is not None
  assert data_fields is not None

  # Store inputs as immutable tuples in this scope, because we close over them
  # for later evaluation. This prevents potentially confusing behavior if the
  # caller were to pass in lists that are later mutated.
  meta_fields = tuple(meta_fields)
  data_fields = tuple(data_fields)

  if dataclasses.is_dataclass(nodetype):
    init_fields = {f.name for f in dataclasses.fields(nodetype) if f.init}
    init_fields.difference_update(*drop_fields)
    if {*meta_fields, *data_fields} != init_fields:
      msg = (
          "data_fields and meta_fields must include all dataclass fields with"
          " ``init=True`` and only them."
      )
      if missing := init_fields - {*meta_fields, *data_fields}:
        msg += (
            f" Missing fields: {missing}. Add them to drop_fields to suppress"
            " this error."
        )
      if unexpected := {*meta_fields, *data_fields} - init_fields:
        msg += f" Unexpected fields: {unexpected}."
      raise ValueError(msg)

  def flatten_with_keys(x):
    meta = tuple(getattr(x, name) for name in meta_fields)
    data = tuple((GetAttrKey(name), getattr(x, name)) for name in data_fields)
    return data, meta

  def unflatten_func(meta, data):
    meta_args = tuple(zip(meta_fields, meta))
    data_args = tuple(zip(data_fields, data))
    kwargs = dict(meta_args + data_args)
    return nodetype(**kwargs)

  def flatten_func(x):
    meta = tuple(getattr(x, name) for name in meta_fields)
    data = tuple(getattr(x, name) for name in data_fields)
    return data, meta

  default_registry.register_dataclass_node(nodetype, list(data_fields), list(meta_fields))
  none_leaf_registry.register_dataclass_node(nodetype, list(data_fields), list(meta_fields))
  dispatch_registry.register_dataclass_node(nodetype, list(data_fields), list(meta_fields))
  _registry[nodetype] = _RegistryEntry(flatten_func, unflatten_func)
  _registry_with_keypaths[nodetype] = _RegistryWithKeypathsEntry(
      flatten_with_keys, unflatten_func
  )
  return nodetype


register_pytree_with_keys(
    collections.OrderedDict,
    lambda x: (tuple((DictKey(k), x[k]) for k in x.keys()), tuple(x.keys())),
    lambda keys, values: collections.OrderedDict(safe_zip(keys, values)),
)

def _flatten_defaultdict_with_keys(d):
  keys = tuple(sorted(d))
  return tuple((DictKey(k), d[k]) for k in keys), (d.default_factory, keys)

register_pytree_with_keys(
    collections.defaultdict,
    _flatten_defaultdict_with_keys,
    lambda s, values: collections.defaultdict(s[0], safe_zip(s[1], values)),
)


@export
def register_static(cls: type[H]) -> type[H]:
  """Registers `cls` as a pytree with no leaves.

  Instances are treated as static by :func:`jax.jit`, :func:`jax.pmap`, etc. This can
  be an alternative to labeling inputs as static using ``jit``'s ``static_argnums``
  and ``static_argnames`` kwargs, ``pmap``'s ``static_broadcasted_argnums``, etc.

  Args:
    cls: type to be registered as static. Must be hashable, as defined in
      https://docs.python.org/3/glossary.html#term-hashable.

  Returns:
    The input class ``cls`` is returned unchanged after being added to JAX's
    pytree registry. This allows ``register_static`` to be used as a decorator.

  Examples:
    >>> import jax
    >>> @jax.tree_util.register_static
    ... class StaticStr(str):
    ...   pass

    This static string can now be used directly in :func:`jax.jit`-compiled
    functions, without marking the variable static using ``static_argnums``:

    >>> @jax.jit
    ... def f(x, y, s):
    ...   return x + y if s == 'add' else x - y
    ...
    >>> f(1, 2, StaticStr('add'))
    Array(3, dtype=int32, weak_type=True)
  """
  flatten = lambda obj: ((), obj)
  unflatten = lambda obj, empty_iter_children: obj
  register_pytree_with_keys(cls, flatten, unflatten)
  return cls


@export
def tree_flatten_with_path(
    tree: Any, is_leaf: Callable[[Any], bool] | None = None
) -> tuple[list[tuple[KeyPath, Any]], PyTreeDef]:
  """Alias of :func:`jax.tree.flatten_with_path`."""
  return default_registry.flatten_with_path(tree, is_leaf)


@export
def tree_leaves_with_path(
    tree: Any, is_leaf: Callable[[Any], bool] | None = None
) -> list[tuple[KeyPath, Any]]:
  """Alias of :func:`jax.tree.leaves_with_path`."""
  return tree_flatten_with_path(tree, is_leaf)[0]


# generate_key_paths is not exported.
def generate_key_paths(
    tree: Any, is_leaf: Callable[[Any], bool] | None = None
) -> list[tuple[KeyPath, Any]]:
  return tree_leaves_with_path(tree, is_leaf)
_generate_key_paths = generate_key_paths  # alias for backward compat


@export
def tree_map_with_path(f: Callable[..., Any],
                       tree: Any, *rest: Any,
                       is_leaf: Callable[[Any], bool] | None = None) -> Any:
  """Alias of :func:`jax.tree.map_with_path`."""
  keypath_leaves, treedef = tree_flatten_with_path(tree, is_leaf)
  keypath_leaves = list(zip(*keypath_leaves))
  all_keypath_leaves = keypath_leaves + [treedef.flatten_up_to(r) for r in rest]
  return treedef.unflatten(f(*xs) for xs in zip(*all_keypath_leaves))


def _child_keys(pytree: Any) -> KeyPath:
  assert not treedef_is_strict_leaf(tree_structure(pytree))
  return tuple(k for k, _ in flatten_one_level_with_keys(pytree)[0])


def _prefix_error(
    key_path: KeyPath,
    prefix_tree: Any,
    full_tree: Any,
    is_leaf: Callable[[Any], bool] | None = None,
) -> Iterable[Callable[[str], ValueError]]:
  # A leaf is a valid prefix of any tree:
  if treedef_is_strict_leaf(tree_structure(prefix_tree, is_leaf=is_leaf)):
    return

  # The subtrees may disagree because their roots are of different types:
  if type(prefix_tree) != type(full_tree):
    yield lambda name: ValueError(
      "pytree structure error: different types at key path\n"
      f"    {name}{keystr(key_path)}\n"
      f"At that key path, the prefix pytree {name} has a subtree of type\n"
      f"    {type(prefix_tree)}\n"
      f"but at the same key path the full pytree has a subtree of different type\n"
      f"    {type(full_tree)}.")
    return  # don't look for more errors in this subtree

  # Or they may disagree if their roots have different numbers or keys of
  # children. Because both prefix_tree and full_tree have the same type at this
  # point, and because prefix_tree is not a leaf, each can be flattened once:
  prefix_tree_children, prefix_tree_meta = flatten_one_level(prefix_tree)
  full_tree_children, full_tree_meta = flatten_one_level(full_tree)
  prefix_tree_children = tuple(prefix_tree_children)
  full_tree_children = tuple(full_tree_children)
  prefix_tree_keys = _child_keys(prefix_tree)
  full_tree_keys = _child_keys(full_tree)
  # First we check special case types (list and tuple, though if they were
  # pytrees we could check strings and sets here, basically Sequences) so that
  # we can report length disagreement rather than integer keys:
  if isinstance(prefix_tree, (list, tuple)):
    if len(prefix_tree) != len(full_tree):
      ty = type(prefix_tree)
      yield lambda name: ValueError(
          f"pytree structure error: different lengths of {ty.__name__} at key path\n"
          f"    {name}{keystr(key_path)}\n"
          f"At that key path, the prefix pytree {name} has a subtree of type "
          f"{ty.__name__} of length {len(prefix_tree)}, but the full pytree "
          f"has a subtree of the same type but of length {len(full_tree)}.")
      return  # don't look for more errors in this subtree
  else:
    # Next we handle the general case of checking child keys.
    try:
      diff = set(prefix_tree_keys).symmetric_difference(set(full_tree_keys))
    except:
      diff = None
    if len(prefix_tree_children) != len(full_tree_children):
      yield lambda name: ValueError(
        "pytree structure error: different numbers of pytree children at key path\n"
        f"    {name}{keystr(key_path)}\n"
        f"At that key path, the prefix pytree {name} has a subtree of type\n"
        f"    {type(prefix_tree)}\n"
        f"with {len(prefix_tree_children)} child keys\n"
        f"    {' '.join(str(k.key) for k in prefix_tree_keys)}\n"
        f"but at the same key path the full pytree has a subtree of the same "
        f"type but with {len(full_tree_children)} child keys\n"
        f"    {' '.join(str(k.key) for k in full_tree_keys)}\n"
        + ("" if diff is None else
           f"so the symmetric difference on key sets is\n"
           f"    {' '.join(str(k.key) for k in diff)}"))
      return  # don't look for more errors in this subtree

  # Or they may disagree if their roots have different pytree metadata:
  if prefix_tree_meta != full_tree_meta:
    prefix_tree_meta_str = str(prefix_tree_meta)
    full_tree_meta_str = str(full_tree_meta)
    metadata_diff = textwrap.indent(
        "\n".join(
            difflib.ndiff(prefix_tree_meta_str.splitlines(),
                          full_tree_meta_str.splitlines())),
        prefix="    ")
    yield lambda name: ValueError(
      "pytree structure error: different pytree metadata at key path\n"
      f"    {name}{keystr(key_path)}\n"
      f"At that key path, the prefix pytree {name} has a subtree of type\n"
      f"    {type(prefix_tree)}\n"
      f"with metadata\n"
      f"    {prefix_tree_meta_str}\n"
      f"but at the same key path the full pytree has a subtree of the same "
      f"type but with metadata\n"
      f"    {full_tree_meta_str}\n"
      f"so the diff in the metadata at these pytree nodes is\n"
      f"{metadata_diff}")
    return  # don't look for more errors in this subtree

  # If the root types and numbers of children agree, there must be an error
  # in a subtree, so recurse:
  assert prefix_tree_keys == full_tree_keys, \
    ("equal pytree nodes gave differing prefix_tree_keys: "
     f"{prefix_tree_keys} and {full_tree_keys}")
  for k, t1, t2 in zip(prefix_tree_keys, prefix_tree_children, full_tree_children):
    yield from _prefix_error((*key_path, k), t1, t2)
