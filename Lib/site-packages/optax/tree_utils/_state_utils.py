# Copyright 2019 DeepMind Technologies Limited. All Rights Reserved.
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
# ==============================================================================
"""Tools for mapping over optimizer states."""

from collections.abc import Callable
import dataclasses
import functools
import typing
from typing import Any, Optional, Protocol, Tuple, Union, cast

import jax
from optax._src import base


@jax.tree_util.register_pytree_node_class
class _ParamsPlaceholder:

  def tree_flatten(self):
    return ((), None)

  @classmethod
  def tree_unflatten(cls, aux, children):
    del aux, children
    return cls()


@dataclasses.dataclass(frozen=True)
class NamedTupleKey:
  """KeyType for a NamedTuple in a tree.

  When using a function ``filtering(path: KeyPath, value: Any) -> bool: ...``
  in a tree in :func:`optax.tree_utils.tree_get_all_with_path`,
  :func:`optax.tree_utils.tree_get`, or :func:`optax.tree_utils.tree_set`, can
  filter the path to check if of the KeyEntry is a NamedTupleKey and then check
  if the name of named tuple is the one intended to be searched.

  Attributes:
    tuple_name (str): name of the tuple containing the key.
    name (str): name of the key.

  .. seealso:: :class:`jax.tree_util.DictKey`,
    :class:`jax.tree_util.FlattenedIndexKey`,
    :class:`jax.tree_util.GetAttrKey`,
    :class:`jax.tree_util.SequenceKey`,
    :func:`optax.tree_utils.tree_get_all_with_path`,
    :func:`optax.tree_utils.tree_get`,
    :func:`optax.tree_utils.tree_set`,

  .. versionadded:: 0.2.2
  """

  tuple_name: str
  name: str

  def __str__(self):
    return f"{self.tuple_name}.{self.name}"


_KeyEntry = Union[
    jax.tree_util.DictKey,
    jax.tree_util.FlattenedIndexKey,
    jax.tree_util.GetAttrKey,
    jax.tree_util.SequenceKey,
    NamedTupleKey,
]

_KeyPath = Tuple[_KeyEntry, ...]


@typing.runtime_checkable
class Initable(Protocol):
  """An object with an init function."""

  def init(self, params: base.Params) -> base.OptState:
    """Calling the init for given parameters returns a fresh opt state."""


def tree_map_params(
    initable: Union[
        Callable[[base.Params], base.OptState],
        Initable,
    ],
    f: Callable[..., Any],
    state: base.OptState,
    /,
    *rest: Any,
    transform_non_params: Optional[Callable[..., Any]] = None,
    is_leaf: Optional[Callable[[base.Params], bool]] = None,
) -> base.OptState:
  """Apply a callable over all params in the given optimizer state.

  This function exists to help construct partition specs over optimizer
  states, in the case that a partition spec is already known for the parameters.

  For example, the following will replace all optimizer state parameter trees
  with copies of the given partition spec instead. The argument
  `transform_non_params` can be used to replace any remaining fields as
  required, in this case, we replace those fields by None.

  >>> params, specs = jnp.array(0.), jnp.array(0.)  # Trees with the same shape
  >>> opt = optax.sgd(1e-3)
  >>> state = opt.init(params)
  >>> opt_specs = optax.tree_map_params(
  ...     opt,
  ...     lambda _, spec: spec,
  ...     state,
  ...     specs,
  ...     transform_non_params=lambda _: None,
  ...     )

  Args:
    initable: A callable taking parameters and returning an optimizer state, or
      an object with an `init` attribute having the same function.
    f: A callable that will be applied for all copies of the parameter tree
      within this optimizer state.
    state: The optimizer state to map over.
    *rest: Additional arguments, having the same shape as the parameter tree,
      that will be passed to f.
    transform_non_params: An optional function that will be called on all
      non-parameter fields within the optimizer state.
    is_leaf: Passed through to `jax.tree.map`. This makes it possible to ignore
      parts of the parameter tree e.g. when the gradient transformations modify
      the shape of the original pytree, such as for ``optax.masked``.

  Returns:
    The result of applying the function f on all trees in the optimizer's state
    that have the same shape as the parameter tree, along with the given
    optional extra arguments.
  """

  # Cast for pytype checks (no-op for other usages).
  placeholder = cast(base.chex.ArrayTree, _ParamsPlaceholder())

  if isinstance(initable, Initable):
    initable = cast(Initable, initable)  # for pytype checks
    state_with_placeholders = initable.init(placeholder)
  else:
    state_with_placeholders = initable(placeholder)

  def map_params(maybe_placeholder_value, value):
    if isinstance(maybe_placeholder_value, _ParamsPlaceholder):
      return jax.tree.map(f, value, *rest, is_leaf=is_leaf)
    elif transform_non_params is not None:
      return transform_non_params(value)
    else:
      return value

  return jax.tree.map(
      map_params,
      state_with_placeholders,
      state,
      is_leaf=lambda v: isinstance(v, _ParamsPlaceholder),
  )


def tree_get_all_with_path(
    tree: base.PyTree,
    key: Any,
    filtering: Optional[Callable[[_KeyPath, Any], bool]] = None,
) -> list[tuple[_KeyPath, Any]]:
  # pylint: disable=line-too-long
  r"""Extract values of a pytree matching a given key.

  Search in a pytree ``tree`` for a specific ``key`` (which can be a key
  from a dictionary, a field from a NamedTuple or the name of a NamedTuple).

  That key/field ``key`` may appear more than once in ``tree``. So this function
  returns a list of all values corresponding to ``key`` with the path to
  that value. The path is a sequence of ``KeyEntry`` that can be transformed in
  readable format using :func:`jax.tree_util.keystr`, see the example below.

  Args:
    tree: tree to search in.
    key: keyword or field to search in tree for.
    filtering: optional callable to further filter values in tree that match the
      key. ``filtering(path: Key_Path, value: Any) -> bool: ...`` takes as
      arguments both the path to the value (as returned by
      :func:`optax.tree_utils.tree_get_all_with_path`) and the value that match
      the given key.

  Returns:
    values_with_path
      list of tuples where each tuple is of the form
      (``path_to_value``, ``value``). Here ``value`` is one entry of the tree
      that corresponds to the ``key``, and ``path_to_value`` is a tuple of
      `KeyEntry` that is a tuple of :class:`jax.tree_util.DictKey`,
      :class:`jax.tree_util.FlattenedIndexKey`,
      :class:`jax.tree_util.GetAttrKey`,
      :class:`jax.tree_util.SequenceKey`, or
      :class:`optax.tree_utils.NamedTupleKey`.

  Examples:

    Basic usage

      >>> import jax.numpy as jnp
      >>> import optax
      >>> params = jnp.array([1., 2., 3.])
      >>> solver = optax.inject_hyperparams(optax.sgd)(
      ...   learning_rate=lambda count: 1/(count+1)
      ... )
      >>> state = solver.init(params)
      >>> found_values_with_path = optax.tree_utils.tree_get_all_with_path(
      ...   state, 'learning_rate'
      ... )
      >>> print(
      ... *[(jax.tree_util.keystr(p), v) for p, v in found_values_with_path],
      ... sep="\n",
      ... )
      ("InjectStatefulHyperparamsState.hyperparams['learning_rate']", Array(1., dtype=float32))
      ("InjectStatefulHyperparamsState.hyperparams_states['learning_rate']", WrappedScheduleState(count=Array(0, dtype=int32)))

    Usage with a filtering operation

      >>> import jax.numpy as jnp
      >>> import optax
      >>> params = jnp.array([1., 2., 3.])
      >>> solver = optax.inject_hyperparams(optax.sgd)(
      ...   learning_rate=lambda count: 1/(count+1)
      ... )
      >>> state = solver.init(params)
      >>> filtering = lambda path, value: isinstance(value, tuple)
      >>> found_values_with_path = optax.tree_utils.tree_get_all_with_path(
      ...   state, 'learning_rate', filtering
      ... )
      >>> print(
      ... *[(jax.tree_util.keystr(p), v) for p, v in found_values_with_path],
      ... sep="\n",
      ... )
      ("InjectStatefulHyperparamsState.hyperparams_states['learning_rate']", WrappedScheduleState(count=Array(0, dtype=int32)))

  .. seealso:: :func:`optax.tree_utils.tree_get`,
    :func:`optax.tree_utils.tree_set`

  .. versionadded:: 0.2.2
  """
  # pylint: enable=line-too-long
  found_values_with_path = _tree_get_all_with_path(tree, key)
  if filtering:
    found_values_with_path = [
        (path, value)
        for path, value in found_values_with_path
        if filtering(path, value)
    ]
  return found_values_with_path


def tree_get(
    tree: base.PyTree,
    key: Any,
    default: Optional[Any] = None,
    filtering: Optional[Callable[[_KeyPath, Any], bool]] = None,
) -> Any:
  # pylint: disable=line-too-long
  """Extract a value from a pytree matching a given key.

  Search in the ``tree`` for a specific ``key`` (which can be a key
  from a dictionary, a field from a NamedTuple or the name of a NamedTuple).

  If the ``tree`` does not containt ``key`` returns ``default``.

  Raises a ``KeyError`` if multiple values of ``key`` are found in ``tree``.

  Generally, you may first get all pairs ``(path_to_value, value)`` for a given
  ``key`` using :func:`optax.tree_utils.tree_get_all_with_path`. You may then
  define a filtering operation
  ``filtering(path: Key_Path, value: Any) -> bool: ...`` that enables you to
  select the specific values you wanted to fetch by looking at the type of the
  value, or looking at the path to that value.
  Note that contrarily to the paths returned by
  :func:`jax.tree_util.tree_leaves_with_path` the paths analyzed by the
  filtering operation in :func:`optax.tree_utils.tree_get_all_with_path`,
  :func:`optax.tree_utils.tree_get`, or :func:`optax.tree_utils.tree_set` detail
  the names of the named tuples considered in the path. Concretely, if the value
  considered is in the attribute ``key`` of a named tuple called
  ``MyNamedTuple`` the last element of the path will be a
  :class:`optax.tree_utils.NamedTupleKey` containing both ``name=key`` and
  ``tuple_name='MyNamedTuple'``. That way you may distinguish between identical
  values in different named tuples (arising for example when chaining
  transformations in optax). See the last example below.

  Args:
    tree: tree to search in.
    key: keyword or field to search in ``tree`` for.
    default: default value to return if ``key`` is not found in ``tree``.
    filtering: optional callable to further filter values in ``tree`` that match
      the ``key``. ``filtering(path: Key_Path, value: Any) -> bool: ...`` takes
      as arguments both the path to the value (as returned by
      :func:`optax.tree_utils.tree_get_all_with_path`) and the value that match
      the given key.

  Returns:
    value
      value in ``tree`` matching the given ``key``. If none are
      found return ``default`` value. If multiple are found raises an error.

  Raises:
    KeyError: If multiple values of ``key`` are found in ``tree``.

  Examples:

    Basic usage

      >>> import jax.numpy as jnp
      >>> import optax
      >>> params = jnp.array([1., 2., 3.])
      >>> opt = optax.adam(learning_rate=1.)
      >>> state = opt.init(params)
      >>> count = optax.tree_utils.tree_get(state, 'count')
      >>> print(count)
      0

    Usage with a filtering operation

      >>> import jax.numpy as jnp
      >>> import optax
      >>> params = jnp.array([1., 2., 3.])
      >>> opt = optax.inject_hyperparams(optax.sgd)(
      ...   learning_rate=lambda count: 1/(count+1)
      ... )
      >>> state = opt.init(params)
      >>> filtering = lambda path, value: isinstance(value, jnp.ndarray)
      >>> lr = optax.tree_utils.tree_get(
      ...   state, 'learning_rate', filtering=filtering
      ... )
      >>> print(lr)
      1.0

    Extracting a named tuple by its name

      >>> params = jnp.array([1., 2., 3.])
      >>> opt = optax.chain(
      ...     optax.add_noise(1.0, 0.9, 0), optax.scale_by_adam()
      ... )
      >>> state = opt.init(params)
      >>> noise_state = optax.tree_utils.tree_get(state, 'AddNoiseState')
      >>> print(noise_state)
      AddNoiseState(count=Array(0, dtype=int32), rng_key=Array([0, 0], dtype=uint32))

    Differentiating between two values by the name of their named tuples.

      >>> import jax.numpy as jnp
      >>> import optax
      >>> params = jnp.array([1., 2., 3.])
      >>> opt = optax.chain(
      ...   optax.add_noise(1.0, 0.9, 0), optax.scale_by_adam()
      ... )
      >>> state = opt.init(params)
      >>> filtering = (
      ...      lambda p, v: isinstance(p[-1], optax.tree_utils.NamedTupleKey)
      ...      and p[-1].tuple_name == 'ScaleByAdamState'
      ... )
      >>> count = optax.tree_utils.tree_get(state, 'count', filtering=filtering)
      >>> print(count)
      0

  .. seealso:: :func:`optax.tree_utils.tree_get_all_with_path`,
    :func:`optax.tree_utils.tree_set`

  .. versionadded:: 0.2.2
  """
  # pylint: enable=line-too-long
  found_values_with_path = tree_get_all_with_path(
      tree, key, filtering=filtering
  )
  if len(found_values_with_path) > 1:
    raise KeyError(f"Found multiple values for '{key}' in {tree}.")
  elif not found_values_with_path:
    return default
  else:
    return found_values_with_path[0][1]


def tree_set(
    tree: base.PyTree,
    filtering: Optional[Callable[[_KeyPath, Any], bool]] = None,
    /,
    **kwargs: Any,
) -> base.PyTree:
  # pylint: disable=line-too-long
  r"""Creates a copy of tree with some values replaced as specified by kwargs.

  Search in the ``tree`` for ``keys`` in ``**kwargs`` (which can be a key
  from a dictionary, a field from a NamedTuple or the name of a NamedTuple).
  If such a key is found, replace the corresponding value with the one given in
  ``**kwargs``.

  Raises a ``KeyError`` if some keys in ``**kwargs`` are not present in the
  tree.

  Args:
    tree: pytree whose values are to be replaced.
    filtering: optional callable to further filter values in ``tree`` that match
      the keys to replace. ``filtering(path: Key_Path, value: Any) -> bool:
      ...`` takes as arguments both the path to the value (as returned by
      :func:`optax.tree_utils.tree_get_all_with_path`) and the value that match
      a given key.
    **kwargs: dictionary of keys with values to replace in ``tree``.

  Returns:
    new_tree
      new pytree with the same structure as ``tree``. For each element in
      ``tree`` whose key/field matches a key in ``**kwargs``, its value is
      set by the corresponding value in ``**kwargs``.

  Raises:
    KeyError: If no values of some key in ``**kwargs`` are found in ``tree``
      or none of the values satisfy the filtering operation.

  Examples:

    Basic usage

      >>> import jax.numpy as jnp
      >>> import optax
      >>> params = jnp.array([1., 2., 3.])
      >>> opt = optax.adam(learning_rate=1.)
      >>> state = opt.init(params)
      >>> print(state)
      (ScaleByAdamState(count=Array(0, dtype=int32), mu=Array([0., 0., 0.], dtype=float32), nu=Array([0., 0., 0.], dtype=float32)), EmptyState())
      >>> new_state = optax.tree_utils.tree_set(state, count=2.)
      >>> print(new_state)
      (ScaleByAdamState(count=2.0, mu=Array([0., 0., 0.], dtype=float32), nu=Array([0., 0., 0.], dtype=float32)), EmptyState())

    Usage with a filtering operation

      >>> import jax.numpy as jnp
      >>> import optax
      >>> params = jnp.array([1., 2., 3.])
      >>> opt = optax.inject_hyperparams(optax.sgd)(
      ...     learning_rate=lambda count: 1/(count+1)
      ...  )
      >>> state = opt.init(params)
      >>> print(state)
      InjectStatefulHyperparamsState(count=Array(0, dtype=int32), hyperparams={'learning_rate': Array(1., dtype=float32)}, hyperparams_states={'learning_rate': WrappedScheduleState(count=Array(0, dtype=int32))}, inner_state=(EmptyState(), EmptyState()))
      >>> filtering = lambda path, value: isinstance(value, jnp.ndarray)
      >>> new_state = optax.tree_utils.tree_set(
      ...   state, filtering, learning_rate=jnp.asarray(0.1)
      ... )
      >>> print(new_state)
      InjectStatefulHyperparamsState(count=Array(0, dtype=int32), hyperparams={'learning_rate': Array(0.1, dtype=float32, weak_type=True)}, hyperparams_states={'learning_rate': WrappedScheduleState(count=Array(0, dtype=int32))}, inner_state=(EmptyState(), EmptyState()))

  .. note:: The recommended usage to inject hyperparameters schedules is through
    :func:`optax.inject_hyperparams`. This function is a helper for other
    purposes.

  .. seealso:: :func:`optax.tree_utils.tree_get_all_with_path`,
    :func:`optax.tree_utils.tree_get`

  .. versionadded:: 0.2.2
  """
  # pylint: enable=line-too-long

  # First check if the keys are present in the tree
  for key in kwargs:
    found_values_with_path = tree_get_all_with_path(tree, key, filtering)
    if not found_values_with_path:
      if filtering:
        raise KeyError(
            f"Found no values matching '{key}' given the filtering operation in"
            f" {tree}"
        )
      else:
        raise KeyError(f"Found no values matching '{key}' in {tree}")

  has_any_key = functools.partial(_node_has_keys, keys=tuple(kwargs.keys()))

  def _replace(path: _KeyPath, node: Any) -> Any:
    """Replace a node with a new node whose values are updated."""
    if has_any_key(node):
      if (
          _is_named_tuple(node)
          and (node.__class__.__name__ in kwargs)
          and (filtering is None or filtering(path, node))
      ):
        # The node itself is a named tuple we wanted to replace
        return kwargs[node.__class__.__name__]
      else:
        # The node contains one of the keys we want to replace
        children_with_path = _get_children_with_path(path, node)
        new_children_with_keys = {}
        for child_path, child in children_with_path:
          # Scan each child of that node
          key = _get_key(child_path[-1])
          if key in kwargs and (
              filtering is None or filtering(child_path, child)
          ):
            # If the child matches a given key given the filtering operation
            # replaces with the new value
            new_children_with_keys.update({key: kwargs[key]})
          else:
            if (
                isinstance(child, tuple)
                or isinstance(child, dict)
                or isinstance(child, list)
            ):
              # If the child is itself a pytree, further search in the child to
              # replace the given value
              new_children_with_keys.update({key: _replace(child_path, child)})
            else:
              # If the child is just a leaf that does not contain the key or
              # satisfies the filtering operation, just return the child.
              new_children_with_keys.update({key: child})
        return _set_children(node, new_children_with_keys)
    else:
      return node

  # Mimics jax.tree_util.tree_map_with_path(_replace, tree, is_leaf)
  # except that the paths we consider can contain NamedTupleKeys
  _, treedef = jax.tree.flatten(tree, is_leaf=has_any_key)
  tree_leaves_with_path = _tree_leaves_with_named_tuple_path(
      tree, is_leaf=has_any_key
  )
  tree_leaves_with_path = list(zip(*tree_leaves_with_path))
  new_tree = treedef.unflatten(
      _replace(*xs) for xs in zip(*tree_leaves_with_path)
  )
  return new_tree


def _tree_get_all_with_path(
    tree: base.PyTree, key: str
) -> list[tuple[_KeyPath, Any]]:
  """Get all values of a pytree matching a given key.

  Private function called recursively, see
  :func:`optax.tree_utils.tree_get_all_with_path` for public api.

  Args:
    tree: tree to search in.
    key: keyword or name to search in tree for.

  Returns:
    values_with_path
      list of tuples where each tuple is of the form
      (``path_to_value``, ``value``). Here ``value`` is one entry of the tree
      that corresponds to the ``key``, and ``path_to_value`` is a tuple of
      `KeyEntry` that is a tuple of :class:`jax.tree_util.DictKey`,
      :class:`jax.tree_util.FlattenedIndexKey`,
      :class:`jax.tree_util.GetAttrKey`,
      :class:`jax.tree_util.SequenceKey`, or
      :class:`optax.tree_utils.NamedTupleKey`.
  """

  # Get subtrees containing a field with the given key
  has_key = functools.partial(_node_has_keys, keys=(key,))
  leaves_or_subtrees_with_path = _tree_leaves_with_named_tuple_path(
      tree, is_leaf=has_key
  )
  subtrees_with_path = [
      (path, leaf_or_subtree)
      for path, leaf_or_subtree in leaves_or_subtrees_with_path
      if has_key(leaf_or_subtree)
  ]

  # Get (path_to_value, value) for the subtrees found
  found_values_with_path = [
      _flatten_to_key(path, subtree, key)
      for path, subtree in subtrees_with_path
  ]

  # Further search in subtrees for additional values
  for path, subtree in subtrees_with_path:
    children_with_path = _get_children_with_path(path, subtree)
    for path, child in children_with_path:
      new_values_with_path = _tree_get_all_with_path(child, key)
      new_values_with_path = [
          ((*path, *new_path), new_value)
          for new_path, new_value in new_values_with_path
      ]
      found_values_with_path += new_values_with_path
  return found_values_with_path


def _is_named_tuple(x):
  return (
      isinstance(x, tuple)
      and hasattr(x, "_fields")
      and hasattr(x, "__class__")
      and hasattr(x.__class__, "__name__")
  )


def _tree_leaves_with_named_tuple_path(
    tree: base.PyTree,
    is_leaf: Optional[
        Callable[
            [
                Any,
            ],
            bool,
        ]
    ] = None,
) -> list[tuple[_KeyPath, Any]]:
  """Get leaves of a tree with their path.

  Essentially the same as :func:`jax.tree_util.tree_leaves_with_path`.
  The difference is that for each attribute of a named tuple we add to the given
  entry the name of the tuple. This facilitates getting/setting values in a
  pytree by filtering for attributes in specific states (different named tuples)
  that have otherwise the same name and type.
  See :func:`optax.tree_utils.tree_get` for a concrete example.

  Args:
    tree: pytree to extract leaves of.
    is_leaf: callable to stop expanding the tree at a node that satisfies
      is_leaf(node) == True.

  Returns:
    list of (path_to_leaf, leaf) for all leaves in the tree
    (or nodes satisfying is_leaf(node) == True).
  """
  is_leaf_ = is_leaf if is_leaf else lambda _: False
  tree_leaves_with_path = jax.tree_util.tree_leaves_with_path(
      tree, is_leaf=lambda x: is_leaf_(x) or _is_named_tuple(x)
  )
  named_tree_leaves_with_path = []
  for path, node in tree_leaves_with_path:
    if is_leaf_(node) or not _is_named_tuple(node):
      named_tree_leaves_with_path.append((path, node))
    else:
      for field in node._fields:
        child_leaves_with_path = _tree_leaves_with_named_tuple_path(
            getattr(node, field), is_leaf
        )
        child_leaves_with_path = [
            (
                (
                    *path,
                    NamedTupleKey(node.__class__.__name__, field),
                    *child_path,
                ),
                child_value,
            )
            for child_path, child_value in child_leaves_with_path
        ]
        named_tree_leaves_with_path = (
            named_tree_leaves_with_path + child_leaves_with_path
        )
  return named_tree_leaves_with_path


def _node_has_keys(node: Any, keys: tuple[Any, ...]) -> bool:
  """Filter for nodes in a tree whose field/key/name matches the given key.

  Private method used in :func:`optax.tree_utils.tree_get_all_with_path` and in
  :func:`optax.tree_utils.tree_set`.

  Args:
    node: node in a pytree.
    keys: keys to search for in the node.

  Returns:
    whether the node has one of the given keys.
  """
  if _is_named_tuple(node) and any(key in node._fields for key in keys):
    return True
  elif _is_named_tuple(node) and (node.__class__.__name__ in keys):
    return True
  elif isinstance(node, dict) and any(key in node for key in keys):
    return True
  else:
    return False


def _flatten_to_key(
    path: _KeyPath, node: Any, key: Any
) -> tuple[_KeyPath, Any]:
  """Flatten a node with a field/key/name matching given key.

  Private method used in :func:`optax.tree_utils.tree_get_all_with_path`.

  Args:
    path: path to the node in a pytree.
    node: node in a pytree.
    key: key to reach for in the node.

  Returns:
    (path_to_key, key_node)
      if key is a key/field of the node,
      ``path_to_key = (*path_to_node, key_path)``, ``key_node = node[key]``,
      otherwise returns the path and node as they are.
  """
  if _is_named_tuple(node):
    if key == node.__class__.__name__:
      return (path, node)
    else:
      path_to_key = (*path, NamedTupleKey(node.__class__.__name__, key))
      return (path_to_key, getattr(node, key))
  elif isinstance(node, dict) and key in node:
    return (*path, jax.tree_util.DictKey(key)), node[key]
  else:
    return path, node


def _get_children_with_path(
    path: _KeyPath, node: Any
) -> list[tuple[_KeyPath, Any]]:
  """Get children of a node.

  Private method used in :func:`optax.tree_utils.tree_get_all_with_path` and in
  :func:`optax.tree_utils.tree_set`. In particular, it is tailored for
  nodes that are NamedTuple or dict.

  Args:
    path: path to the node in a pytree.
    node: node in a pytree.

  Returns:
    list of (path_to_child, child) for child a child in nodes.

  Raises:
    ValueError if the given node is not a NamedTuple or a dict
  """
  if _is_named_tuple(node):
    return [
        (
            (*path, NamedTupleKey(node.__class__.__name__, field)),
            getattr(node, field),
        )
        for field in node._fields
    ]
  elif isinstance(node, dict):
    return [
        ((*path, jax.tree_util.DictKey(key)), value)
        for key, value in node.items()
    ]
  else:
    raise ValueError(
        f"Subtree must be a dict or a NamedTuple. Got {type(node)}"
    )


def _set_children(node: Any, children_with_keys: dict[Any, Any]) -> Any:
  """Set children of a node.

  Private method used in :func:`optax.tree_utils.tree_set`.
  In particular, it is tailored for nodes that are NamedTuple or dict.

  Args:
    node: node in a pytree.
    children_with_keys: children of the node with associated keys

  Returns:
    new_node whose fields/keys are replaced by the ones given in
    children_with_keys.

  Raises:
    ValueError if the given node is not a NamedTuple or a dict
  """
  if _is_named_tuple(node):
    return node._replace(**children_with_keys)
  elif isinstance(node, dict):
    return children_with_keys
  else:
    raise ValueError(
        f"Subtree must be a dict or a NamedTuple. Got {type(node)}"
    )


def _get_key(key: _KeyEntry) -> Union[int, str]:
  """Convert a ``KeyEntry``` to a usual type."""
  if isinstance(key, jax.tree_util.DictKey):
    if isinstance(key.key, (str, int)):
      return key.key
    raise KeyError("Hashable keys not supported")
  if isinstance(key, jax.tree_util.FlattenedIndexKey):
    return key.key  # int.
  if isinstance(key, jax.tree_util.GetAttrKey):
    return key.name  # str.
  if isinstance(key, jax.tree_util.SequenceKey):
    return key.idx  # int.
  if isinstance(key, NamedTupleKey):
    return key.name  # str.
  raise KeyError(f"Tree key '{key}' of type '{type(key)}' not valid.")
