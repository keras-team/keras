from keras.src.api_export import keras_export
from keras.src.utils.module_utils import dmtree
from keras.src.utils.module_utils import optree

if optree.available:
    from keras.src.tree import optree_impl as tree_impl
elif dmtree.available:
    from keras.src.tree import dmtree_impl as tree_impl
else:
    raise ImportError(
        "To use Keras, you need to have `optree` installed. "
        "Install it via `pip install optree`"
    )


def register_tree_node_class(cls):
    return tree_impl.register_tree_node_class(cls)


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
    return tree_impl.is_nested(structure)


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
    return tree_impl.traverse(func, structure, top_down=top_down)


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
    return tree_impl.flatten(structure)


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
    return tree_impl.map_structure(func, *structures)


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
    return tree_impl.map_structure_up_to(shallow_structure, func, *structures)


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
    return tree_impl.assert_same_structure(a, b, check_types=check_types)


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
    return tree_impl.pack_sequence_as(
        structure, flat_sequence, sequence_fn=sequence_fn
    )


@keras_export("keras.tree.lists_to_tuples")
def lists_to_tuples(structure):
    return tree_impl.lists_to_tuples(structure)


@keras_export("keras.tree.map_shape_structure")
def map_shape_structure(func, structure):
    """Variant of keras.tree.map_structure that operates on shape tuples."""
    return tree_impl.map_shape_structure(func, structure)
