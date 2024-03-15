import collections
import collections.abc
import types

import optree

from keras.api_export import keras_export
from keras.backend.config import backend

# Register backend-specific node classes
if backend() == "tensorflow":
    from tensorflow.python.trackable.data_structures import ListWrapper

    optree.register_pytree_node(
        ListWrapper,
        lambda x: (x, None),
        lambda metadata, children: ListWrapper(list(children)),
        namespace="keras",
    )


@keras_export("keras.tree.is_nested")
def is_nested(structure):
    """Checks if a given structure is nested.

    Examples:

    >>> keras.tree.is_nested(42)
    False
    >>> keras.tree.is_nested({"foo": 42})
    True

    Args:
        structure: A structure to check.

    Returns:
        `True` if a given structure is nested, i.e. is a sequence, a mapping,
        or a namedtuple, and `False` otherwise.
    """
    return not optree.tree_is_leaf(
        structure, none_is_leaf=True, namespace="keras"
    )


@keras_export("keras.tree.traverse")
def traverse(func, structure, top_down=True):
    """Traverses the given nested structure, applying the given function.

    The traversal is depth-first. If `top_down` is True (default), parents
    are returned before their children (giving the option to avoid traversing
    into a sub-tree).

    Examples:

    >>> v = []
    >>> keras.tree.traverse(v.append, [(1, 2), [3], {"a": 4}], top_down=True)
    [(1, 2), [3], {'a': 4}]
    >>> v
    [[(1, 2), [3], {'a': 4}], (1, 2), 1, 2, [3], 3, {'a': 4}, 4]

    >>> v = []
    >>> keras.tree.traverse(v.append, [(1, 2), [3], {"a": 4}], top_down=False)
    [(1, 2), [3], {'a': 4}]
    >>> v
    [1, 2, (1, 2), 3, [3], 4, {'a': 4}, [(1, 2), [3], {'a': 4}]]

    Args:
        func: The function to be applied to each sub-nest of the structure.

        When traversing top-down:
            If `func(subtree) is None` the traversal continues into the
            sub-tree.
            If `func(subtree) is not None` the traversal does not continue
            into the sub-tree. The sub-tree will be replaced by `func(subtree)`
            in the returned structure (to replace the sub-tree with `None`, use
            the special value `_MAP_TO_NONE`).

        When traversing bottom-up:
            If `func(subtree) is None` the traversed sub-tree is returned
            unaltered.
            If `func(subtree) is not None` the sub-tree will be replaced by
            `func(subtree)` in the returned structure (to replace the sub-tree
            with None, use the special value `_MAP_TO_NONE`).

        structure: The structure to traverse.
        top_down: If True, parent structures will be visited before their
            children.

    Returns:
        The structured output from the traversal.
    """

    # From https://github.com/google/jax/pull/19695
    def traverse_children():
        children, treedef = optree.tree_flatten(
            structure,
            is_leaf=lambda x: x is not structure,
            none_is_leaf=True,
            namespace="keras",
        )
        if treedef.num_nodes == 1 and treedef.num_leaves == 1:
            return structure
        else:
            return optree.tree_unflatten(
                treedef,
                [traverse(func, c, top_down=top_down) for c in children],
            )

    if top_down:
        ret = func(structure)
        if ret is None:
            return traverse_children()
    else:
        traversed_structure = traverse_children()
        ret = func(traversed_structure)
        if ret is None:
            return traversed_structure
    return None if ret is _MAP_TO_NONE else ret


@keras_export("keras.tree.flatten")
def flatten(structure):
    """Flattens a possibly nested structure into a list.

    In the case of dict instances, the sequence consists of the values,
    sorted by key to ensure deterministic behavior. This is true also for
    `collections.OrderedDict` instances: their sequence order is
    considered. The same convention is followed in `unflatten_as`.
    This correctly unflattens dicts and `OrderedDict` after they have been
    flattened, or vice-versa.

    Dictionaries with non-sortable keys cannot be flattened.

    Examples:

    >>> keras.tree.flatten([[1, 2, 3], [4, [5], [[6]]]])
    [1, 2, 3, 4, 5, 6]
    >>> keras.tree.flatten(None)
    [None]
    >>> keras.tree.flatten(1)
    [1]
    >>> keras.tree.flatten({100: 'world!', 6: 'Hello'})
    ['Hello', 'world!']

    Args:
        structure: An arbitrarily nested structure.

    Returns:
        A list, the flattened version of the input `structure`.
    """
    # optree.tree_flatten returns a pair (leaves, treespec) where the first
    # element is a list of leaf values and the second element is a treespec
    # representing the structure of the pytree.
    leaves, _ = optree.tree_flatten(
        structure, none_is_leaf=True, namespace="keras"
    )
    return leaves


@keras_export("keras.tree.unflatten_as")
def unflatten_as(structure, flat_sequence):
    """Unflattens a sequence into a given structure.

    If `structure` is a scalar, `flat_sequence` must be a single-element list;
    in this case the return value is ``flat_sequence[0]``.

    If `structure` is or contains a dict instance, the keys will be sorted to
    pack the flat sequence in deterministic order. This is true also for
    `collections.OrderedDict` instances: their sequence order is considered.
    The same convention is followed in `flatten`. This correctly unflattens
    dicts and `OrderedDict` after they have been flattened, or vice-versa.

    Dictionaries with non-sortable keys cannot be unflattened.

    Examples:

    >>> keras.tree.unflatten_as([[1, 2], [[3], [4]]], [5, 6, 7, 8])
    [[5, 6], [[7], [8]]]
    >>> keras.tree.unflatten_as(None, [1])
    1
    >>> keras.tree.unflatten_as({1: None, 2: None}, ['Hello', 'world!'])
    {1: 'Hello', 2: 'world!'}

    Args:
        structure: Arbitrarily nested structure.
        flat_sequence: Sequence to unflatten.

    Returns:
        `flat_sequence` unflattened into `structure`.
    """
    if not is_nested(flat_sequence):
        raise TypeError(
            f"flat_sequence must be a sequence not a {type(flat_sequence)}:\n"
            f"{flat_sequence}"
        )
    if not is_nested(structure):
        if len(flat_sequence) != 1:
            raise ValueError(
                "Structure is a scalar but "
                f"len(flat_sequence) == {len(flat_sequence)} > 1"
            )
        return flat_sequence[0]
    structure_spec = optree.tree_structure(
        structure, none_is_leaf=True, namespace="keras"
    )
    return structure_spec.unflatten(flat_sequence)


@keras_export("keras.tree.map_structure")
def map_structure(func, *structures):
    """Maps `func` through given structures.

    Examples:

    >>> structure = [[1], [2], [3]]
    >>> keras.tree.map_structure(lambda v: v**2, structure)
    [[1], [4], [9]]
    >>> keras.tree.map_structure(lambda x, y: x * y, structure, structure)
    [[1], [4], [9]]

    >>> Foo = collections.namedtuple('Foo', ['a', 'b'])
    >>> structure = Foo(a=1, b=2)
    >>> keras.tree.map_structure(lambda v: v * 2, structure)
    Foo(a=2, b=4)

    Args:
        func: A callable that accepts as many arguments as there are structures.
        *structures: Arbitrarily nested structures of the same layout.

    Returns:
        A new structure with the same layout as the given ones.
    """
    if not callable(func):
        raise TypeError(f"`func` must be callable. Received: func={func}")
    if not structures:
        raise ValueError("Must provide at least one structure")
    for other in structures[1:]:
        assert_same_structure(structures[0], other, check_types=False)
    return optree.tree_map(
        func, *structures, none_is_leaf=True, namespace="keras"
    )


@keras_export("keras.tree.map_structure_up_to")
def map_structure_up_to(shallow_structure, func, *structures):
    """Maps `func` through given structures up to `shallow_structure`.

    This is a variant of `map_structure` which only maps the given structures
    up to `shallow_structure`. All further nested components are retained as-is.

    Examples:

    >>> shallow_structure = [None, None]
    >>> structure = [[1, 1], [2, 2]]
    >>> keras.tree.map_structure_up_to(shallow_structure, len, structure)
    [2, 2]

    >>> shallow_structure = [None, [None, None]]
    >>> keras.tree.map_structure_up_to(shallow_structure, str, structure)
    ['[1, 1]', ['2', '2']]

    Args:
        shallow_structure: A structure with layout common to all `structures`.
        func: A callable that accepts as many arguments as there are structures.
        *structures: Arbitrarily nested structures of the same layout.

    Returns:
        A new structure with the same layout as `shallow_structure`.
    """
    return _map_structure_with_path_up_to(
        shallow_structure,
        lambda _, *args: func(*args),  # Discards path.
        *structures,
    )


@keras_export("keras.tree.assert_same_structure")
def assert_same_structure(a, b, check_types=True):
    """Asserts that two structures are nested in the same way.

    Note that namedtuples with identical name and fields will not be considered
    as same structures even `check_types=False`.

    Examples:

    >>> keras.tree.assert_same_structure([(0, 1)], [(2, 3)])

    >>> Foo = collections.namedtuple('Foo', ['a', 'b'])
    >>> AlsoFoo = collections.namedtuple('Foo', ['a', 'b'])
    >>> keras.tree.assert_same_structure(Foo(0, 1), Foo(2, 3))
    >>> keras.tree.assert_same_structure(Foo(0, 1), AlsoFoo(2, 3))
    Traceback (most recent call last):
        ...
    ValueError: `a` and `b` don't have the same structure.
    ...

    Args:
        a: an arbitrarily nested structure.
        b: an arbitrarily nested structure.
        check_types: if `True` (default) types of leaves are checked as well.
    """
    a_structure = optree.tree_structure(a, none_is_leaf=True, namespace="keras")
    b_structure = optree.tree_structure(b, none_is_leaf=True, namespace="keras")
    if a_structure != b_structure:
        raise ValueError(
            "`a` and `b` don't have the same structure. "
            f"Received: structure of a={a_structure}, "
            f"structure of b={b_structure}"
        )
    if check_types:
        type_structure = optree.tree_map(
            lambda x, y: type(x) is type(y),
            a,
            b,
            none_is_leaf=True,
            namespace="keras",
        )
        if not optree.tree_all(
            type_structure, none_is_leaf=True, namespace="keras"
        ):
            raise TypeError(
                "The type of the leaves of `a` and `b` doesn't match."
            )


@keras_export("keras.tree.pack_sequence_as")
def pack_sequence_as(structure, flat_sequence, sequence_fn=None):
    """Returns a given flattened sequence packed into a given structure.

    If `structure` is an atom, `flat_sequence` must be a single-item list; in
    this case the return value is `flat_sequence[0]`.

    If `structure` is or contains a dict instance, the keys will be sorted to
    pack the flat sequence in deterministic order. This is true also for
    `OrderedDict` instances: their sequence order is considered. The same
    convention is followed in `flatten`. This correctly repacks dicts and
    `OrderedDicts` after they have been flattened, or vice-versa.

    Dictionaries with non-sortable keys cannot be flattened.

    Examples:

    >>> structure = {"key3": "", "key1": "", "key2": ""}
    >>> flat_sequence = ["value1", "value2", "value3"]
    >>> keras.tree.pack_sequence_as(structure, flat_sequence)
    {"key3": "value3", "key1": "value1", "key2": "value2"}

    >>> structure = (("a", "b"), ("c", "d", "e"), "f")
    >>> flat_sequence = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    >>> keras.tree.pack_sequence_as(structure, flat_sequence)
    ((1.0, 2.0), (3.0, 4.0, 5.0), 6.0)

    >>> structure = {"key3": {"c": ("alpha", "beta"), "a": ("gamma")},
    ... "key1": {"e": "val1", "d": "val2"}}
    >>> flat_sequence = ["val2", "val1", 3.0, 1.0, 2.0]
    >>> keras.tree.pack_sequence_as(structure, flat_sequence)
    {'key3': {'c': (1.0, 2.0), 'a': 3.0}, 'key1': {'e': 'val1', 'd': 'val2'}}

    >>> structure = ["a"]
    >>> flat_sequence = [np.array([[1, 2], [3, 4]])]
    >>> keras.tree.pack_sequence_as(structure, flat_sequence)
    [array([[1, 2],
       [3, 4]])]

    >>> structure = ["a"]
    >>> flat_sequence = [keras.ops.ones([2, 2])]
    >>> keras.tree.pack_sequence_as(structure, flat_sequence)
    [array([[1., 1.],
       [1., 1.]]]

    Args:
        structure: Arbitrarily nested structure.
        flat_sequence: Flat sequence to pack.
        sequence_fn: Defaults to `_sequence_like`.

    Returns:
        `flat_sequence` converted to have the same recursive structure as
        `structure`.
    """
    sequence_fn = sequence_fn or _sequence_like

    def truncate(value, length):
        value_str = str(value)
        return value_str[:length] + (value_str[length:] and "...")

    if not is_nested(flat_sequence):
        raise TypeError(
            "Attempted to pack value:\n  {}\ninto a structure, but found "
            "incompatible type `{}` instead.".format(
                truncate(flat_sequence, 100), type(flat_sequence)
            )
        )

    if not is_nested(structure):
        if len(flat_sequence) != 1:
            raise ValueError(
                "The target structure is of type `{}`\n  {}\nHowever the input "
                "is a sequence ({}) of length {}.\n  {}\nnest cannot "
                "guarantee that it is safe to map one to the other.".format(
                    type(structure),
                    truncate(structure, 100),
                    type(flat_sequence),
                    len(flat_sequence),
                    truncate(flat_sequence, 100),
                )
            )
        return flat_sequence[0]

    try:
        final_index, packed = _packed_nest_with_indices(
            structure, flat_sequence, 0, sequence_fn
        )
        if final_index < len(flat_sequence):
            raise IndexError
    except IndexError:
        flat_structure = flatten(structure)
        if len(flat_structure) != len(flat_sequence):
            # pylint: disable=raise-missing-from
            raise ValueError(
                "Could not pack sequence. "
                f"Structure had {len(flat_structure)} atoms, but "
                f"flat_sequence had {len(flat_sequence)} items. "
                f"Structure: {structure}, flat_sequence: {flat_sequence}."
            )
    return sequence_fn(structure, packed)


@keras_export("keras.tree.lists_to_tuples")
def lists_to_tuples(structure):
    """Converts `list`s to `tuple`s."""

    def sequence_fn(instance, args):
        if isinstance(instance, list):
            return tuple(args)
        return _sequence_like(instance, args)

    return pack_sequence_as(
        structure, flatten(structure), sequence_fn=sequence_fn
    )


class _MapToNone:
    """A special object used as a sentinel within `traverse`."""

    def __repr__(self):
        return "keras.utils.tree._MAP_TO_NONE"


_MAP_TO_NONE = _MapToNone()


def _yield_flat_up_to(shallow_tree, input_tree, path=()):
    if isinstance(shallow_tree, (str, bytes)) or not (
        isinstance(
            shallow_tree, (collections.abc.Mapping, collections.abc.Sequence)
        )
        or optree.is_namedtuple(shallow_tree)
    ):
        yield (path, input_tree)
    else:
        input_tree = dict(_yield_sorted_items(input_tree))
        for shallow_key, shallow_subtree in _yield_sorted_items(shallow_tree):
            subpath = path + (shallow_key,)
            input_subtree = input_tree[shallow_key]
            for leaf_path, leaf_value in _yield_flat_up_to(
                shallow_subtree, input_subtree, path=subpath
            ):
                yield (leaf_path, leaf_value)


def _multiyield_flat_up_to(shallow_tree, *input_trees):
    """Same as `_yield_flat_up_to`, but takes multiple input trees."""
    zipped_iterators = zip(
        *[
            _yield_flat_up_to(shallow_tree, input_tree)
            for input_tree in input_trees
        ]
    )
    try:
        for paths_and_values in zipped_iterators:
            paths, values = zip(*paths_and_values)
            yield paths[:1] + values
    except KeyError as e:
        paths = locals().get("paths", ((),))
        raise ValueError(
            f"Could not find key '{e.args[0]}' in some `input_trees`. "
            "Please ensure the structure of all `input_trees` are "
            "compatible with `shallow_tree`. The last valid path "
            f"yielded was {paths[0]}."
        ) from e


def _map_structure_with_path_up_to(shallow_structure, func, *structures):
    results = []
    for path_and_values in _multiyield_flat_up_to(
        shallow_structure, *structures
    ):
        results.append(func(*path_and_values))
    shallow_structure_spec = optree.tree_structure(
        shallow_structure, none_is_leaf=True, namespace="keras"
    )
    return shallow_structure_spec.unflatten(results)


def _sequence_like(instance, args):
    # TODO: Support attrs library
    if isinstance(instance, (dict, collections.abc.Mapping)):
        # Pack dictionaries in a deterministic order by sorting the keys.
        # Notice this means that we ignore the original order of `OrderedDict`
        # instances. This is intentional, to avoid potential bugs caused by
        # mixing ordered and plain dicts (e.g., flattening a dict but using a
        # corresponding `OrderedDict` to pack it back).
        result = dict(zip(sorted(instance), args))
        keys_and_values = ((key, result[key]) for key in instance)
        if isinstance(instance, collections.defaultdict):
            # `defaultdict` requires a default factory as the first argument.
            return type(instance)(instance.default_factory, keys_and_values)
        elif isinstance(instance, types.MappingProxyType):
            # MappingProxyType requires a dict to proxy to.
            return type(instance)(dict(keys_and_values))
        else:
            return type(instance)(keys_and_values)
    elif isinstance(instance, collections.abc.MappingView):
        # We can't directly construct mapping views, so we create a list instead
        return list(args)
    elif optree.is_namedtuple(instance):
        instance_type = type(instance)
        try:
            return instance_type(*args)
        except Exception as e:
            raise TypeError(
                f"Couldn't traverse {instance!r} with arguments {args}"
            ) from e
    else:
        # Not a namedtuple
        return type(instance)(args)


def _yield_sorted_items(iterable):
    # TODO: Support attrs library
    if isinstance(iterable, collections.abc.Mapping):
        # Iterate through dictionaries in a deterministic order by sorting the
        # keys. Notice this means that we ignore the original order of
        # `OrderedDict` instances. This is intentional, to avoid potential bugs
        # caused by mixing ordered and plain dicts (e.g., flattening a dict but
        # using a corresponding `OrderedDict` to pack it back).
        for key in sorted(iterable):
            yield key, iterable[key]
    elif optree.is_namedtuple(iterable):
        for field in iterable._fields:
            yield (field, getattr(iterable, field))
    else:
        for item in enumerate(iterable):
            yield item


def _yield_value(iterable):
    for _, v in _yield_sorted_items(iterable):
        yield v


def _packed_nest_with_indices(structure, flat, index, sequence_fn=None):
    packed = []
    sequence_fn = sequence_fn or _sequence_like
    for s in _yield_value(structure):
        if is_nested(s):
            new_index, child = _packed_nest_with_indices(
                s, flat, index, sequence_fn
            )
            packed.append(sequence_fn(s, child))
            index = new_index
        else:
            packed.append(flat[index])
            index += 1
    return index, packed
