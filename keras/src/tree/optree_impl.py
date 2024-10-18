import collections
import collections.abc
import types

import optree
import optree.utils

from keras.src.backend.config import backend


def register_tree_node_class(cls):
    return optree.register_pytree_node_class(cls, namespace="keras")


# Register backend-specific node classes
if backend() == "tensorflow":
    from tensorflow.python.trackable.data_structures import ListWrapper

    optree.register_pytree_node(
        ListWrapper,
        lambda x: (x, None),
        lambda metadata, children: ListWrapper(list(children)),
        namespace="keras",
    )

    from tensorflow.python.trackable.data_structures import _DictWrapper

    optree.register_pytree_node(
        _DictWrapper,
        lambda x: (list(x.values()), list(x.keys())),
        lambda metadata, children: _DictWrapper(
            {key: child for key, child in zip(metadata, children)}
        ),
        namespace="keras",
    )


def is_nested(structure):
    return not optree.tree_is_leaf(
        structure, none_is_leaf=True, namespace="keras"
    )


def traverse(func, structure, top_down=True):
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


def flatten(structure):
    # optree.tree_flatten returns a pair (leaves, treespec) where the first
    # element is a list of leaf values and the second element is a treespec
    # representing the structure of the pytree.
    leaves, _ = optree.tree_flatten(
        structure, none_is_leaf=True, namespace="keras"
    )
    return leaves


def map_structure(func, *structures):
    if not callable(func):
        raise TypeError(f"`func` must be callable. Received: func={func}")
    if not structures:
        raise ValueError("Must provide at least one structure")
    for other in structures[1:]:
        assert_same_structure(structures[0], other, check_types=False)
    return optree.tree_map(
        func, *structures, none_is_leaf=True, namespace="keras"
    )


def map_structure_up_to(shallow_structure, func, *structures):
    return _map_structure_with_path_up_to(
        shallow_structure,
        lambda _, *args: func(*args),  # Discards path.
        *structures,
    )


def assert_same_structure(a, b, check_types=True):
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


def pack_sequence_as(structure, flat_sequence, sequence_fn=None):
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


def lists_to_tuples(structure):

    def sequence_fn(instance, args):
        if isinstance(instance, list):
            return tuple(args)
        return _sequence_like(instance, args)

    return pack_sequence_as(
        structure, flatten(structure), sequence_fn=sequence_fn
    )


def map_shape_structure(func, structure):

    def is_shape_tuple(x):
        return isinstance(x, (list, tuple)) and all(
            isinstance(e, (int, type(None))) for e in x
        )

    if not callable(func):
        raise TypeError(f"`func` must be callable. Received: func={func}")
    return optree.tree_map(
        func,
        structure,
        is_leaf=is_shape_tuple,
        none_is_leaf=True,
        namespace="keras",
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
