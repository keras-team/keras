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

"""Functions for working with nested data structures."""

from collections import abc as collections_abc
import logging
import sys
from typing import Mapping, Sequence, TypeVar, Union

from .sequence import _is_attrs
from .sequence import _is_namedtuple
from .sequence import _sequence_like
from .sequence import _sorted

# pylint: disable=g-import-not-at-top
try:
  import wrapt
  ObjectProxy = wrapt.ObjectProxy
except ImportError:
  class ObjectProxy(object):
    """Stub-class for `wrapt.ObjectProxy``."""

try:
  from tree import _tree
except ImportError:
  if "sphinx" not in sys.modules:
    raise

  _tree = None

# pylint: enable=g-import-not-at-top

__all__ = [
    "is_nested",
    "assert_same_structure",
    "unflatten_as",
    "flatten",
    "flatten_up_to",
    "flatten_with_path",
    "flatten_with_path_up_to",
    "map_structure",
    "map_structure_up_to",
    "map_structure_with_path",
    "map_structure_with_path_up_to",
    "traverse",
    "MAP_TO_NONE",
]

__version__ = "0.1.9"

# Note: this is *not* the same as `six.string_types`, which in Python3 is just
#       `(str,)` (i.e. it does not include byte strings).
_TEXT_OR_BYTES = (str, bytes)

_SHALLOW_TREE_HAS_INVALID_KEYS = (
    "The shallow_tree's keys are not a subset of the input_tree's keys. The "
    "shallow_tree has the following keys that are not in the input_tree: {}.")

_STRUCTURES_HAVE_MISMATCHING_TYPES = (
    "The two structures don't have the same sequence type. Input structure has "
    "type {input_type}, while shallow structure has type {shallow_type}.")

_STRUCTURES_HAVE_MISMATCHING_LENGTHS = (
    "The two structures don't have the same sequence length. Input "
    "structure has length {input_length}, while shallow structure has length "
    "{shallow_length}."
)

_INPUT_TREE_SMALLER_THAN_SHALLOW_TREE = (
    "The input_tree has fewer elements than the shallow_tree. Input structure "
    "has length {input_size}, while shallow structure has length "
    "{shallow_size}.")

_IF_SHALLOW_IS_SEQ_INPUT_MUST_BE_SEQ = (
    "If shallow structure is a sequence, input must also be a sequence. "
    "Input has type: {}.")

_IF_SHALLOW_IS_SEQ_INPUT_MUST_BE_SEQ_WITH_PATH = (
    "If shallow structure is a sequence, input must also be a sequence. "
    "Input at path: {path} has type: {input_type}.")

K = TypeVar("K")
V = TypeVar("V")

# A generic monomorphic structure type, e.g. ``StructureKV[str, int]``
# is an arbitrarily nested structure where keys must be of type ``str``
# and values are integers.
StructureKV = Union[
    Sequence["StructureKV[K, V]"],
    Mapping[K, "StructureKV[K, V]"],
    V,
]
Structure = StructureKV[str, V]


def _get_attrs_items(obj):
  """Returns a list of (name, value) pairs from an attrs instance.

  The list will be sorted by name.

  Args:
    obj: an object.

  Returns:
    A list of (attr_name, attr_value) pairs.
  """
  return [(attr.name, getattr(obj, attr.name))
          for attr in obj.__class__.__attrs_attrs__]


def _yield_value(iterable):
  for _, v in _yield_sorted_items(iterable):
    yield v


def _yield_sorted_items(iterable):
  """Yield (key, value) pairs for `iterable` in a deterministic order.

  For Sequences, the key will be an int, the array index of a value.
  For Mappings, the key will be the dictionary key.
  For objects (e.g. namedtuples), the key will be the attribute name.

  In all cases, the keys will be iterated in sorted order.

  Args:
    iterable: an iterable.

  Yields:
    The iterable's (key, value) pairs, in order of sorted keys.
  """
  if isinstance(iterable, collections_abc.Mapping):
    # Iterate through dictionaries in a deterministic order by sorting the
    # keys. Notice this means that we ignore the original order of `OrderedDict`
    # instances. This is intentional, to avoid potential bugs caused by mixing
    # ordered and plain dicts (e.g., flattening a dict but using a
    # corresponding `OrderedDict` to pack it back).
    for key in _sorted(iterable):
      yield key, iterable[key]
  elif _is_attrs(iterable):
    for item in _get_attrs_items(iterable):
      yield item
  elif _is_namedtuple(iterable):
    for field in iterable._fields:
      yield (field, getattr(iterable, field))
  else:
    for item in enumerate(iterable):
      yield item


def _num_elements(structure):
  if _is_attrs(structure):
    return len(getattr(structure.__class__, "__attrs_attrs__"))
  else:
    return len(structure)


def is_nested(structure):
  """Checks if a given structure is nested.

  >>> tree.is_nested(42)
  False
  >>> tree.is_nested({"foo": 42})
  True

  Args:
    structure: A structure to check.

  Returns:
    `True` if a given structure is nested, i.e. is a sequence, a mapping,
    or a namedtuple, and `False` otherwise.
  """
  return _tree.is_sequence(structure)


def flatten(structure):
  r"""Flattens a possibly nested structure into a list.

  >>> tree.flatten([[1, 2, 3], [4, [5], [[6]]]])
  [1, 2, 3, 4, 5, 6]

  If `structure` is not nested, the result is a single-element list.

  >>> tree.flatten(None)
  [None]
  >>> tree.flatten(1)
  [1]

  In the case of dict instances, the sequence consists of the values,
  sorted by key to ensure deterministic behavior. This is true also for
  :class:`~collections.OrderedDict` instances: their sequence order is
  ignored, the sorting order of keys is used instead. The same convention
  is followed in :func:`~tree.unflatten`. This correctly unflattens dicts
  and ``OrderedDict``\ s after they have been flattened, and also allows
  flattening an ``OrderedDict`` and then unflattening it back using a
  corresponding plain dict, or vice-versa.

  Dictionaries with non-sortable keys cannot be flattened.

  >>> tree.flatten({100: 'world!', 6: 'Hello'})
  ['Hello', 'world!']

  Args:
    structure: An arbitrarily nested structure.

  Returns:
    A list, the flattened version of the input `structure`.

  Raises:
    TypeError: If `structure` is or contains a mapping with non-sortable keys.
  """
  return _tree.flatten(structure)


class _DotString(object):

  def __str__(self):
    return "."

  def __repr__(self):
    return "."


_DOT = _DotString()


def assert_same_structure(a, b, check_types=True):
  """Asserts that two structures are nested in the same way.

  >>> tree.assert_same_structure([(0, 1)], [(2, 3)])

  Note that namedtuples with identical name and fields are always considered
  to have the same shallow structure (even with `check_types=True`).

  >>> Foo = collections.namedtuple('Foo', ['a', 'b'])
  >>> AlsoFoo = collections.namedtuple('Foo', ['a', 'b'])
  >>> tree.assert_same_structure(Foo(0, 1), AlsoFoo(2, 3))

  Named tuples with different names are considered to have different shallow
  structures:

  >>> Bar = collections.namedtuple('Bar', ['a', 'b'])
  >>> tree.assert_same_structure(Foo(0, 1), Bar(2, 3))
  Traceback (most recent call last):
    ...
  TypeError: The two structures don't have the same nested structure.
  ...

  Args:
    a: an arbitrarily nested structure.
    b: an arbitrarily nested structure.
    check_types: if `True` (default) types of sequences are checked as
      well, including the keys of dictionaries. If set to `False`, for example
      a list and a tuple of objects will look the same if they have the same
      size. Note that namedtuples with identical name and fields are always
      considered to have the same shallow structure.

  Raises:
    ValueError: If the two structures do not have the same number of elements or
      if the two structures are not nested in the same way.
    TypeError: If the two structures differ in the type of sequence in any of
      their substructures. Only possible if `check_types` is `True`.
  """
  try:
    _tree.assert_same_structure(a, b, check_types)
  except (ValueError, TypeError) as e:
    str1 = str(map_structure(lambda _: _DOT, a))
    str2 = str(map_structure(lambda _: _DOT, b))
    raise type(e)("%s\n"
                  "Entire first structure:\n%s\n"
                  "Entire second structure:\n%s"
                  % (e, str1, str2))


def _packed_nest_with_indices(structure, flat, index):
  """Helper function for ``unflatten_as``.

  Args:
    structure: Substructure (list / tuple / dict) to mimic.
    flat: Flattened values to output substructure for.
    index: Index at which to start reading from flat.

  Returns:
    The tuple (new_index, child), where:
      * new_index - the updated index into `flat` having processed `structure`.
      * packed - the subset of `flat` corresponding to `structure`,
                 having started at `index`, and packed into the same nested
                 format.

  Raises:
    ValueError: if `structure` contains more elements than `flat`
      (assuming indexing starts from `index`).
  """
  packed = []
  for s in _yield_value(structure):
    if is_nested(s):
      new_index, child = _packed_nest_with_indices(s, flat, index)
      packed.append(_sequence_like(s, child))
      index = new_index
    else:
      packed.append(flat[index])
      index += 1
  return index, packed


def unflatten_as(structure, flat_sequence):
  r"""Unflattens a sequence into a given structure.

  >>> tree.unflatten_as([[1, 2], [[3], [4]]], [5, 6, 7, 8])
  [[5, 6], [[7], [8]]]

  If `structure` is a scalar, `flat_sequence` must be a single-element list;
  in this case the return value is ``flat_sequence[0]``.

  >>> tree.unflatten_as(None, [1])
  1

  If `structure` is or contains a dict instance, the keys will be sorted to
  pack the flat sequence in deterministic order. This is true also for
  :class:`~collections.OrderedDict` instances: their sequence order is
  ignored, the sorting order of keys is used instead. The same convention
  is followed in :func:`~tree.flatten`. This correctly unflattens dicts
  and ``OrderedDict``\ s after they have been flattened, and also allows
  flattening an ``OrderedDict`` and then unflattening it back using a
  corresponding plain dict, or vice-versa.

  Dictionaries with non-sortable keys cannot be unflattened.

  >>> tree.unflatten_as({1: None, 2: None}, ['Hello', 'world!'])
  {1: 'Hello', 2: 'world!'}

  Args:
    structure: Arbitrarily nested structure.
    flat_sequence: Sequence to unflatten.

  Returns:
    `flat_sequence` unflattened into `structure`.

  Raises:
    ValueError: If `flat_sequence` and `structure` have different
      element counts.
    TypeError: If `structure` is or contains a mapping with non-sortable keys.
  """
  if not is_nested(flat_sequence):
    raise TypeError("flat_sequence must be a sequence not a {}:\n{}".format(
        type(flat_sequence), flat_sequence))

  if not is_nested(structure):
    if len(flat_sequence) != 1:
      raise ValueError("Structure is a scalar but len(flat_sequence) == %d > 1"
                       % len(flat_sequence))
    return flat_sequence[0]

  flat_structure = flatten(structure)
  if len(flat_structure) != len(flat_sequence):
    raise ValueError(
        "Could not pack sequence. Structure had %d elements, but flat_sequence "
        "had %d elements.  Structure: %s, flat_sequence: %s."
        % (len(flat_structure), len(flat_sequence), structure, flat_sequence))

  _, packed = _packed_nest_with_indices(structure, flat_sequence, 0)
  return _sequence_like(structure, packed)


def map_structure(func, *structures, **kwargs):  # pylint: disable=redefined-builtin
  """Maps `func` through given structures.

  >>> structure = [[1], [2], [3]]
  >>> tree.map_structure(lambda v: v**2, structure)
  [[1], [4], [9]]
  >>> tree.map_structure(lambda x, y: x * y, structure, structure)
  [[1], [4], [9]]
  >>> Foo = collections.namedtuple('Foo', ['a', 'b'])
  >>> structure = Foo(a=1, b=2)
  >>> tree.map_structure(lambda v: v * 2, structure)
  Foo(a=2, b=4)

  Args:
    func: A callable that accepts as many arguments as there are structures.
    *structures: Arbitrarily nested structures of the same layout.
    **kwargs: The only valid keyword argument is `check_types`. If `True`
      (default) the types of components within the structures have
      to be match, e.g. ``tree.map_structure(func, [1], (1,))`` will raise
      a `TypeError`, otherwise this is not enforced. Note that namedtuples
      with identical name and fields are considered to be the same type.

  Returns:
    A new structure with the same layout as the given ones. If the
    `structures` have components of varying types, the resulting structure
    will use the same types as ``structures[0]``.

  Raises:
    TypeError: If `func` is not callable.
    ValueError: If the two structures do not have the same number of elements or
      if the two structures are not nested in the same way.
    TypeError: If `check_types` is `True` and any two `structures`
      differ in the types of their components.
    ValueError: If no structures were given or if a keyword argument other
      than `check_types` is provided.
  """
  if not callable(func):
    raise TypeError("func must be callable, got: %s" % func)

  if not structures:
    raise ValueError("Must provide at least one structure")

  check_types = kwargs.pop("check_types", True)
  if kwargs:
    raise ValueError(
        "Only valid keyword arguments are `check_types` "
        "not: `%s`" % ("`, `".join(kwargs.keys())))

  for other in structures[1:]:
    assert_same_structure(structures[0], other, check_types=check_types)
  return unflatten_as(structures[0],
                      [func(*args) for args in zip(*map(flatten, structures))])


def map_structure_with_path(func, *structures, **kwargs):
  """Maps `func` through given structures.

  This is a variant of :func:`~tree.map_structure` which accumulates
  a *path* while mapping through the structures. A path is a tuple of
  indices and/or keys which uniquely identifies the positions of the
  arguments passed to `func`.

  >>> tree.map_structure_with_path(
  ...     lambda path, v: (path, v**2),
  ...     [{"foo": 42}])
  [{'foo': ((0, 'foo'), 1764)}]

  Args:
    func: A callable that accepts a path and as many arguments as there are
      structures.
    *structures: Arbitrarily nested structures of the same layout.
    **kwargs: The only valid keyword argument is `check_types`. If `True`
      (default) the types of components within the structures have to be match,
      e.g. ``tree.map_structure_with_path(func, [1], (1,))`` will raise a
      `TypeError`, otherwise this is not enforced. Note that namedtuples with
      identical name and fields are considered to be the same type.

  Returns:
    A new structure with the same layout as the given ones. If the
    `structures` have components of varying types, the resulting structure
    will use the same types as ``structures[0]``.

  Raises:
    TypeError: If `func` is not callable or if the `structures` do not
      have the same layout.
    TypeError: If `check_types` is `True` and any two `structures`
      differ in the types of their components.
    ValueError: If no structures were given or if a keyword argument other
      than `check_types` is provided.
  """
  return map_structure_with_path_up_to(structures[0], func, *structures,
                                       **kwargs)


def _yield_flat_up_to(shallow_tree, input_tree, path=()):
  """Yields (path, value) pairs of input_tree flattened up to shallow_tree.

  Args:
    shallow_tree: Nested structure. Traverse no further than its leaf nodes.
    input_tree: Nested structure. Return the paths and values from this tree.
      Must have the same upper structure as shallow_tree.
    path: Tuple. Optional argument, only used when recursing. The path from the
      root of the original shallow_tree, down to the root of the shallow_tree
      arg of this recursive call.

  Yields:
    Pairs of (path, value), where path the tuple path of a leaf node in
    shallow_tree, and value is the value of the corresponding node in
    input_tree.
  """
  if (isinstance(shallow_tree, _TEXT_OR_BYTES) or
      not (isinstance(shallow_tree, (collections_abc.Mapping,
                                     collections_abc.Sequence)) or
           _is_namedtuple(shallow_tree) or
           _is_attrs(shallow_tree))):
    yield (path, input_tree)
  else:
    input_tree = dict(_yield_sorted_items(input_tree))
    for shallow_key, shallow_subtree in _yield_sorted_items(shallow_tree):
      subpath = path + (shallow_key,)
      input_subtree = input_tree[shallow_key]
      for leaf_path, leaf_value in _yield_flat_up_to(shallow_subtree,
                                                     input_subtree,
                                                     path=subpath):
        yield (leaf_path, leaf_value)


def _multiyield_flat_up_to(shallow_tree, *input_trees):
  """Same as `_yield_flat_up_to`, but takes multiple input trees."""
  zipped_iterators = zip(*[_yield_flat_up_to(shallow_tree, input_tree)
                           for input_tree in input_trees])
  try:
    for paths_and_values in zipped_iterators:
      paths, values = zip(*paths_and_values)
      yield paths[:1] + values
  except KeyError as e:
    paths = locals().get("paths", ((),))
    raise ValueError(f"Could not find key '{e.args[0]}' in some `input_trees`. "
                     "Please ensure the structure of all `input_trees` are "
                     "compatible with `shallow_tree`. The last valid path "
                     f"yielded was {paths[0]}.") from e


def _assert_shallow_structure(shallow_tree,
                              input_tree,
                              path=None,
                              check_types=True):
  """Asserts that `shallow_tree` is a shallow structure of `input_tree`.

  That is, this function recursively tests if each key in shallow_tree has its
  corresponding key in input_tree.

  Examples:

  The following code will raise an exception:

  >>> shallow_tree = {"a": "A", "b": "B"}
  >>> input_tree = {"a": 1, "c": 2}
  >>> _assert_shallow_structure(shallow_tree, input_tree)
  Traceback (most recent call last):
    ...
  ValueError: The shallow_tree's keys are not a subset of the input_tree's ...

  The following code will raise an exception:

  >>> shallow_tree = ["a", "b"]
  >>> input_tree = ["c", ["d", "e"], "f"]
  >>> _assert_shallow_structure(shallow_tree, input_tree)
  Traceback (most recent call last):
    ...
  ValueError: The two structures don't have the same sequence length. ...

  By setting check_types=False, we drop the requirement that corresponding
  nodes in shallow_tree and input_tree have to be the same type. Sequences
  are treated equivalently to Mappables that map integer keys (indices) to
  values. The following code will therefore not raise an exception:

  >>> _assert_shallow_structure({0: "foo"}, ["foo"], check_types=False)

  Args:
    shallow_tree: an arbitrarily nested structure.
    input_tree: an arbitrarily nested structure.
    path: if not `None`, a tuple containing the current path in the nested
      structure. This is only used for more informative errror messages.
    check_types: if `True` (default) the sequence types of `shallow_tree` and
      `input_tree` have to be the same.

  Raises:
    TypeError: If `shallow_tree` is a sequence but `input_tree` is not.
    TypeError: If the sequence types of `shallow_tree` are different from
      `input_tree`. Only raised if `check_types` is `True`.
    ValueError: If the sequence lengths of `shallow_tree` are different from
      `input_tree`.
  """
  if is_nested(shallow_tree):
    if not is_nested(input_tree):
      if path is not None:
        raise TypeError(
            _IF_SHALLOW_IS_SEQ_INPUT_MUST_BE_SEQ_WITH_PATH.format(
                path=list(path), input_type=type(input_tree)))
      else:
        raise TypeError(
            _IF_SHALLOW_IS_SEQ_INPUT_MUST_BE_SEQ.format(
                type(input_tree)))

    if isinstance(shallow_tree, ObjectProxy):
      shallow_type = type(shallow_tree.__wrapped__)
    else:
      shallow_type = type(shallow_tree)

    if check_types and not isinstance(input_tree, shallow_type):
      # Duck-typing means that nest should be fine with two different
      # namedtuples with identical name and fields.
      shallow_is_namedtuple = _is_namedtuple(shallow_tree, False)
      input_is_namedtuple = _is_namedtuple(input_tree, False)
      if shallow_is_namedtuple and input_is_namedtuple:
        # pylint: disable=protected-access
        if not _tree.same_namedtuples(shallow_tree, input_tree):
          raise TypeError(_STRUCTURES_HAVE_MISMATCHING_TYPES.format(
              input_type=type(input_tree),
              shallow_type=shallow_type))
        # pylint: enable=protected-access
      elif not (isinstance(shallow_tree, collections_abc.Mapping)
                and isinstance(input_tree, collections_abc.Mapping)):
        raise TypeError(_STRUCTURES_HAVE_MISMATCHING_TYPES.format(
            input_type=type(input_tree),
            shallow_type=shallow_type))

    if _num_elements(input_tree) != _num_elements(shallow_tree):
      raise ValueError(
          _STRUCTURES_HAVE_MISMATCHING_LENGTHS.format(
              input_length=_num_elements(input_tree),
              shallow_length=_num_elements(shallow_tree)))
    elif _num_elements(input_tree) < _num_elements(shallow_tree):
      raise ValueError(
          _INPUT_TREE_SMALLER_THAN_SHALLOW_TREE.format(
              input_size=_num_elements(input_tree),
              shallow_size=_num_elements(shallow_tree)))

    shallow_iter = _yield_sorted_items(shallow_tree)
    input_iter = _yield_sorted_items(input_tree)

    def get_matching_input_branch(shallow_key):
      for input_key, input_branch in input_iter:
        if input_key == shallow_key:
          return input_branch

      raise ValueError(_SHALLOW_TREE_HAS_INVALID_KEYS.format([shallow_key]))

    for shallow_key, shallow_branch in shallow_iter:
      input_branch = get_matching_input_branch(shallow_key)
      _assert_shallow_structure(
          shallow_branch,
          input_branch,
          path + (shallow_key,) if path is not None else None,
          check_types=check_types)


def flatten_up_to(shallow_structure, input_structure, check_types=True):
  """Flattens `input_structure` up to `shallow_structure`.

  All further nested components in `input_structure` are retained as-is.

  >>> structure = [[1, 1], [2, 2]]
  >>> tree.flatten_up_to([None, None], structure)
  [[1, 1], [2, 2]]
  >>> tree.flatten_up_to([None, [None, None]], structure)
  [[1, 1], 2, 2]

  If `shallow_structure` and `input_structure` are not nested, the
  result is a single-element list:

  >>> tree.flatten_up_to(42, 1)
  [1]
  >>> tree.flatten_up_to(42, [1, 2, 3])
  [[1, 2, 3]]

  Args:
    shallow_structure: A structure with the same (but possibly more shallow)
      layout as `input_structure`.
    input_structure: An arbitrarily nested structure.
    check_types: If `True`, check that each node in shallow_tree has the
      same type as the corresponding node in `input_structure`.

  Returns:
    A list, the partially flattened version of `input_structure` wrt
    `shallow_structure`.

  Raises:
    TypeError: If the layout of `shallow_structure` does not match that of
      `input_structure`.
    TypeError: If `check_types` is `True` and `shallow_structure` and
      `input_structure` differ in the types of their components.
  """
  _assert_shallow_structure(
      shallow_structure, input_structure, path=None, check_types=check_types)
  # Discard paths returned by _yield_flat_up_to.
  return [v for _, v in _yield_flat_up_to(shallow_structure, input_structure)]


def flatten_with_path_up_to(shallow_structure,
                            input_structure,
                            check_types=True):
  """Flattens `input_structure` up to `shallow_structure`.

  This is a combination of :func:`~tree.flatten_up_to` and
  :func:`~tree.flatten_with_path`

  Args:
    shallow_structure: A structure with the same (but possibly more shallow)
      layout as `input_structure`.
    input_structure: An arbitrarily nested structure.
    check_types: If `True`, check that each node in shallow_tree has the
      same type as the corresponding node in `input_structure`.

  Returns:
    A list of ``(path, item)`` pairs corresponding to the partially flattened
    version of `input_structure` wrt `shallow_structure`.

  Raises:
    TypeError: If the layout of `shallow_structure` does not match that of
      `input_structure`.
    TypeError: If `input_structure` is or contains a mapping with non-sortable
      keys.
    TypeError: If `check_types` is `True` and `shallow_structure` and
      `input_structure` differ in the types of their components.
  """
  _assert_shallow_structure(
      shallow_structure, input_structure, path=(), check_types=check_types)
  return list(_yield_flat_up_to(shallow_structure, input_structure))


def map_structure_up_to(shallow_structure, func, *structures, **kwargs):
  """Maps `func` through given structures up to `shallow_structure`.

  This is a variant of :func:`~tree.map_structure` which only maps
  the given structures up to `shallow_structure`. All further nested
  components are retained as-is.

  >>> structure = [[1, 1], [2, 2]]
  >>> tree.map_structure_up_to([None, None], len, structure)
  [2, 2]
  >>> tree.map_structure_up_to([None, [None, None]], str, structure)
  ['[1, 1]', ['2', '2']]

  Args:
    shallow_structure: A structure with layout common to all `structures`.
    func: A callable that accepts as many arguments as there are structures.
    *structures: Arbitrarily nested structures of the same layout.
    **kwargs: No valid keyword arguments.
  Raises:
    ValueError: If `func` is not callable or if `structures` have different
      layout or if the layout of `shallow_structure` does not match that of
      `structures` or if no structures were given.

  Returns:
    A new structure with the same layout as `shallow_structure`.
  """
  return map_structure_with_path_up_to(
      shallow_structure,
      lambda _, *args: func(*args),  # Discards path.
      *structures,
      **kwargs)


def map_structure_with_path_up_to(shallow_structure, func, *structures,
                                  **kwargs):
  """Maps `func` through given structures up to `shallow_structure`.

  This is a combination of :func:`~tree.map_structure_up_to` and
  :func:`~tree.map_structure_with_path`

  Args:
    shallow_structure: A structure with layout common to all `structures`.
    func: A callable that accepts a path and as many arguments as there are
      structures.
    *structures: Arbitrarily nested structures of the same layout.
    **kwargs: No valid keyword arguments.

  Raises:
    ValueError: If `func` is not callable or if `structures` have different
      layout or if the layout of `shallow_structure` does not match that of
      `structures` or if no structures were given.

  Returns:
    Result of repeatedly applying `func`. Has the same structure layout
    as `shallow_tree`.
  """
  if "check_types" in kwargs:
    logging.warning("The use of `check_types` is deprecated and does not have "
                    "any effect.")
  del kwargs
  results = []
  for path_and_values in _multiyield_flat_up_to(shallow_structure, *structures):
    results.append(func(*path_and_values))
  return unflatten_as(shallow_structure, results)


def flatten_with_path(structure):
  r"""Flattens a possibly nested structure into a list.

  This is a variant of :func:`~tree.flattens` which produces a list of
  pairs: ``(path, item)``.  A path is a tuple of indices and/or keys
  which uniquely identifies the position of the corresponding ``item``.

  >>> tree.flatten_with_path([{"foo": 42}])
  [((0, 'foo'), 42)]

  Args:
    structure: An arbitrarily nested structure.

  Returns:
    A list of ``(path, item)`` pairs corresponding to the flattened version
    of the input `structure`.

  Raises:
    TypeError:
      If ``structure`` is or contains a mapping with non-sortable keys.
  """
  return list(_yield_flat_up_to(structure, structure))


#: Special value for use with :func:`traverse`.
MAP_TO_NONE = object()


def traverse(fn, structure, top_down=True):
  """Traverses the given nested structure, applying the given function.

  The traversal is depth-first. If ``top_down`` is True (default), parents
  are returned before their children (giving the option to avoid traversing
  into a sub-tree).

  >>> visited = []
  >>> tree.traverse(visited.append, [(1, 2), [3], {"a": 4}], top_down=True)
  [(1, 2), [3], {'a': 4}]
  >>> visited
  [[(1, 2), [3], {'a': 4}], (1, 2), 1, 2, [3], 3, {'a': 4}, 4]

  >>> visited = []
  >>> tree.traverse(visited.append, [(1, 2), [3], {"a": 4}], top_down=False)
  [(1, 2), [3], {'a': 4}]
  >>> visited
  [1, 2, (1, 2), 3, [3], 4, {'a': 4}, [(1, 2), [3], {'a': 4}]]

  Args:
    fn: The function to be applied to each sub-nest of the structure.

      When traversing top-down:
        If ``fn(subtree) is None`` the traversal continues into the sub-tree.
        If ``fn(subtree) is not None`` the traversal does not continue into
        the sub-tree. The sub-tree will be replaced by ``fn(subtree)`` in the
        returned structure (to replace the sub-tree with None, use the special
        value :data:`MAP_TO_NONE`).

      When traversing bottom-up:
        If ``fn(subtree) is None`` the traversed sub-tree is returned unaltered.
        If ``fn(subtree) is not None`` the sub-tree will be replaced by
        ``fn(subtree)`` in the returned structure (to replace the sub-tree
        with None, use the special value :data:`MAP_TO_NONE`).

    structure: The structure to traverse.
    top_down: If True, parent structures will be visited before their children.

  Returns:
    The structured output from the traversal.
  """
  return traverse_with_path(lambda _, x: fn(x), structure, top_down=top_down)


def traverse_with_path(fn, structure, top_down=True):
  """Traverses the given nested structure, applying the given function.

  The traversal is depth-first. If ``top_down`` is True (default), parents
  are returned before their children (giving the option to avoid traversing
  into a sub-tree).

  >>> visited = []
  >>> tree.traverse_with_path(
  ...  lambda path, subtree: visited.append((path, subtree)),
  ...  [(1, 2), [3], {"a": 4}],
  ...  top_down=True)
  [(1, 2), [3], {'a': 4}]
  >>> visited == [
  ...  ((), [(1, 2), [3], {'a': 4}]),
  ...  ((0,), (1, 2)),
  ...  ((0, 0), 1),
  ...  ((0, 1), 2),
  ...  ((1,), [3]),
  ...  ((1, 0), 3),
  ...  ((2,), {'a': 4}),
  ...  ((2, 'a'), 4)]
  True

  >>> visited = []
  >>> tree.traverse_with_path(
  ...  lambda path, subtree: visited.append((path, subtree)),
  ...  [(1, 2), [3], {"a": 4}],
  ...  top_down=False)
  [(1, 2), [3], {'a': 4}]
  >>> visited == [
  ...  ((0, 0), 1),
  ...  ((0, 1), 2),
  ...  ((0,), (1, 2)),
  ...  ((1, 0), 3),
  ...  ((1,), [3]),
  ...  ((2, 'a'), 4),
  ...  ((2,), {'a': 4}),
  ... ((), [(1, 2), [3], {'a': 4}])]
  True

  Args:
    fn: The function to be applied to the path to each sub-nest of the structure
      and the sub-nest value.
      When traversing top-down: If ``fn(path, subtree) is None`` the traversal
        continues into the sub-tree. If ``fn(path, subtree) is not None`` the
        traversal does not continue into the sub-tree. The sub-tree will be
        replaced by ``fn(path, subtree)`` in the returned structure (to replace
        the sub-tree with None, use the special
        value :data:`MAP_TO_NONE`).
      When traversing bottom-up: If ``fn(path, subtree) is None`` the traversed
        sub-tree is returned unaltered. If ``fn(path, subtree) is not None`` the
        sub-tree will be replaced by ``fn(path, subtree)`` in the returned
        structure (to replace the sub-tree
        with None, use the special value :data:`MAP_TO_NONE`).
    structure: The structure to traverse.
    top_down: If True, parent structures will be visited before their children.

  Returns:
    The structured output from the traversal.
  """

  def traverse_impl(path, structure):
    """Recursive traversal implementation."""

    def subtree_fn(item):
      subtree_path, subtree = item
      return traverse_impl(path + (subtree_path,), subtree)

    def traverse_subtrees():
      if is_nested(structure):
        return _sequence_like(structure,
                              map(subtree_fn, _yield_sorted_items(structure)))
      else:
        return structure

    if top_down:
      ret = fn(path, structure)
      if ret is None:
        return traverse_subtrees()
      elif ret is MAP_TO_NONE:
        return None
      else:
        return ret
    else:
      traversed_structure = traverse_subtrees()
      ret = fn(path, traversed_structure)
      if ret is None:
        return traversed_structure
      elif ret is MAP_TO_NONE:
        return None
      else:
        return ret

  return traverse_impl((), structure)


