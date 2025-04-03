import collections
import collections.abc
import itertools

from keras.src.backend.config import backend
from keras.src.utils.module_utils import dmtree

# NOTE: There are two known discrepancies between this `dmtree` implementation
# of the tree API and the `optree` implementation:
#
# 1. `map_structure` with *multiple* structures and `map_structure_up_to` do not
#    use the object registration (they use the raw `dmtree.map_structure` and
#    `dmtree.map_structure_up_to`). This only has consequences with two types of
#    structures:
#    - `TrackedSet` will not explored (considered as a leaf).
#    - `OrderedDict` will be traversed in the order of sorted keys, not the
#      order of the items. This is typically inconsequential because functions
#      used with `map_structure` and `map_structure_up_to` are typically not
#      order dependent and are, in fact, stateless.
#
# 2. The handling of non-sortable keys in dictionaries in inconsistent. `optree`
#    uses the iteration order while `dmtree` raises an error. This is not an
#    issue as keys are always strings. But this is the reason why we document
#    non-sortable keys as unsupported (meaning behavior is undefined).

REGISTERED_CLASSES = {}

ClassRegistration = collections.namedtuple(
    "ClassRegistration", ["flatten", "unflatten"]
)


class TypeErrorRemapping:
    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type is TypeError:
            raise ValueError(exc_value).with_traceback(traceback)
        return False


def register_tree_node(
    cls,
    flatten_func=None,
    unflatten_func=None,
):
    if flatten_func is None:
        flatten_func = lambda x: x.tree_flatten()
    if unflatten_func is None:
        unflatten_func = cls.tree_unflatten
    REGISTERED_CLASSES[cls] = ClassRegistration(flatten_func, unflatten_func)


def register_tree_node_class(cls):
    register_tree_node(cls)
    return cls


register_tree_node(
    collections.OrderedDict,
    lambda d: (d.values(), list(d.keys()), d.keys()),
    lambda metadata, children: collections.OrderedDict(zip(metadata, children)),
)

if backend() == "tensorflow":
    from tensorflow.python.trackable.data_structures import ListWrapper
    from tensorflow.python.trackable.data_structures import _DictWrapper

    register_tree_node(
        ListWrapper,
        lambda x: (x, None),
        lambda metadata, children: ListWrapper(list(children)),
    )

    def sorted_keys_and_values(d):
        keys = sorted(list(d.keys()))
        values = [d[k] for k in keys]
        return values, keys, keys

    register_tree_node(
        _DictWrapper,
        sorted_keys_and_values,
        lambda metadata, children: _DictWrapper(
            {key: child for key, child in zip(metadata, children)}
        ),
    )


def is_nested(structure):
    return type(structure) in REGISTERED_CLASSES or dmtree.is_nested(structure)


def traverse(func, structure, top_down=True):
    if not callable(func):
        raise TypeError(
            f"`func` must be callable, got {func} of type {type(func)}"
        )

    def remap_map_to_none(value, new_value):
        if isinstance(value, type) and value.__name__ == "MAP_TO_NONE":
            return new_value
        return value

    def traverse_top_down(s):
        ret = func(s)
        if ret is not None:
            return remap_map_to_none(ret, dmtree.MAP_TO_NONE)
        registration = REGISTERED_CLASSES.get(type(s), None)
        if registration is None:
            return None
        flat_meta_s = registration.flatten(s)
        flat_s = [
            dmtree.traverse(traverse_top_down, x, top_down=True)
            for x in list(flat_meta_s[0])
        ]
        return registration.unflatten(flat_meta_s[1], flat_s)

    def traverse_bottom_up(s):
        registration = REGISTERED_CLASSES.get(type(s), None)
        if registration is not None:
            flat_meta_s = registration.flatten(s)
            ret = [traverse_bottom_up(x) for x in list(flat_meta_s[0])]
            ret = registration.unflatten(flat_meta_s[1], ret)
        elif not dmtree.is_nested(s):
            ret = s
        elif isinstance(s, collections.abc.Mapping):
            ret = [traverse_bottom_up(s[key]) for key in sorted(s)]
            ret = dmtree._sequence_like(s, ret)
        else:
            ret = [traverse_bottom_up(x) for x in s]
            ret = dmtree._sequence_like(s, ret)
        func_ret = func(ret)
        return ret if func_ret is None else remap_map_to_none(func_ret, None)

    if top_down:
        return dmtree.traverse(traverse_top_down, structure, top_down=True)
    else:
        return traverse_bottom_up(structure)


def flatten(structure):
    if not is_nested(structure):
        return [structure]

    flattened = []

    def flatten_func(s):
        registration = REGISTERED_CLASSES.get(type(s), None)
        if registration is not None:
            flat_s = list(registration.flatten(s)[0])
            return dmtree.traverse(flatten_func, flat_s, top_down=True)
        if not is_nested(s):
            flattened.append(s)
            return dmtree.MAP_TO_NONE if s is None else s
        return None

    dmtree.traverse(flatten_func, structure, top_down=True)
    return flattened


def _recursive_flatten_with_path(path, structure, flattened):
    registration = REGISTERED_CLASSES.get(type(structure), None)
    if registration is not None:
        flat_meta_paths = registration.flatten(structure)
        flat = flat_meta_paths[0]
        paths = (
            flat_meta_paths[2]
            if len(flat_meta_paths) >= 3
            else itertools.count()
        )
        for key, value in zip(paths, flat):
            _recursive_flatten_with_path(path + (key,), value, flattened)
    elif not dmtree.is_nested(structure):
        flattened.append((path, structure))
    elif isinstance(structure, collections.abc.Mapping):
        for key in sorted(structure):
            _recursive_flatten_with_path(
                path + (key,), structure[key], flattened
            )
    else:
        for key, value in enumerate(structure):
            _recursive_flatten_with_path(path + (key,), value, flattened)


def flatten_with_path(structure):
    if not is_nested(structure):
        return [((), structure)]

    # Fully reimplemented in Python to handle registered classes, OrderedDict
    # and namedtuples the same way as optree.
    flattened = []
    _recursive_flatten_with_path((), structure, flattened)
    return flattened


def map_structure(func, *structures):
    if not callable(func):
        raise TypeError(
            f"`func` must be callable, got {func} of type {type(func)}"
        )

    def func_traverse_wrapper(s):
        if is_nested(s):
            return None
        ret = func(s)
        if ret is None:
            return dmtree.MAP_TO_NONE
        return ret

    if len(structures) == 1:
        return traverse(func_traverse_wrapper, structures[0])

    with TypeErrorRemapping():
        return dmtree.map_structure(func, *structures)


def map_structure_up_to(shallow_structure, func, *structures):
    if not callable(func):
        raise TypeError(
            f"`func` must be callable, got {func} of type {type(func)}"
        )

    with TypeErrorRemapping():
        return dmtree.map_structure_up_to(shallow_structure, func, *structures)


def assert_same_structure(a, b):
    # Fully reimplemented in Python to handle registered classes.

    # Don't handle OrderedDict as a registered class, use the normal dict path
    # so that OrderedDict is equivalent to dict per optree behavior.
    a_registration = REGISTERED_CLASSES.get(type(a), None)
    if isinstance(a, collections.OrderedDict):
        a_registration = None

    b_registration = REGISTERED_CLASSES.get(type(b), None)
    if isinstance(b, collections.OrderedDict):
        b_registration = None

    if a_registration != b_registration:
        raise ValueError(
            f"Custom node type mismatch; "
            f"expected type: {type(a)}, got type: {type(b)} "
            f"while comparing {a} and {b}."
        )
    if a_registration is not None:
        a_flat_meta = a_registration.flatten(a)
        b_flat_meta = b_registration.flatten(b)
        a_flat = list(a_flat_meta[0])
        b_flat = list(b_flat_meta[0])
        if not a_flat_meta[1] == b_flat_meta[1]:
            raise ValueError(
                f"Mismatch custom node data; "
                f"expected: {a_flat_meta[1]}, got: {b_flat_meta[1]} "
                f"while comparing {a} and {b}."
            )
        if len(a_flat) != len(b_flat):
            raise ValueError(
                f"Arity mismatch; expected: {len(a)}, got: {len(b)} "
                f"while comparing {a} and {b}."
            )
        for sub_a, sub_b in zip(a_flat, b_flat):
            assert_same_structure(sub_a, sub_b)
    elif not dmtree.is_nested(a):
        if dmtree.is_nested(b):
            raise ValueError(
                f"Structures don't have the same nested structure: {a}, {b}."
            )
    elif isinstance(
        a, (dict, collections.OrderedDict, collections.defaultdict)
    ):
        if not isinstance(
            b, (dict, collections.OrderedDict, collections.defaultdict)
        ):
            raise ValueError(
                f"Expected an instance of dict, collections.OrderedDict, or "
                f"collections.defaultdict, got {type(b)} "
                f"while comparing {a} and {b}."
            )
        a_keys = sorted(a)
        b_keys = sorted(b)
        if not a_keys == b_keys:
            raise ValueError(
                f"Dictionary key mismatch; "
                f"expected key(s): {a_keys}, got key(s): {b_keys} "
                f"while comparing {a} and {b}."
            )
        for key in a_keys:
            assert_same_structure(a[key], b[key])
    elif isinstance(a, collections.abc.Mapping):
        raise ValueError(
            f"Encountered unregistered collections.abc.Mapping type: {type(a)} "
            f"while comparing {a} and {b}."
        )
    else:
        if type(a) is not type(b):
            raise ValueError(
                f"Expected an instance of {type(a)}, got {type(b)} "
                f"while comparing {a} and {b}."
            )
        if not len(a) == len(b):
            raise ValueError(
                f"Arity mismatch; expected: {len(a)}, got: {len(b)} "
                f"while comparing {a} and {b}."
            )
        for sub_a, sub_b in zip(a, b):
            assert_same_structure(sub_a, sub_b)


def assert_same_paths(a, b):
    a_paths = set([path for path, _ in flatten_with_path(a)])
    b_paths = set([path for path, _ in flatten_with_path(b)])

    if a_paths != b_paths:
        msg = "`a` and `b` don't have the same paths."
        a_diff = a_paths.difference(b_paths)
        if a_diff:
            msg += f"\nPaths in `a` missing in `b`:\n{a_diff}"
        b_diff = b_paths.difference(a_paths)
        if b_diff:
            msg += f"\nPaths in `b` missing in `a`:\n{b_diff}"
        raise ValueError(msg)


def pack_sequence_as(structure, flat_sequence):
    # This is not just an optimization for the case when structure is a leaf.
    # This is required to avoid Torch Dynamo failures.
    if not is_nested(structure):
        if len(flat_sequence) == 1:
            return flat_sequence[0]
        else:
            raise ValueError(
                "Incorrect number of leaves provided by `flat_sequence` for "
                f"`structure`; expected: 1, got {len(flat_sequence)}."
            )

    flat_sequence_it = enumerate(flat_sequence)

    def unflatten_func(s):
        registration = REGISTERED_CLASSES.get(type(s), None)
        if registration is not None:
            flat_meta_s = registration.flatten(s)
            flat_s = dmtree.traverse(
                unflatten_func, list(flat_meta_s[0]), top_down=True
            )
            return registration.unflatten(flat_meta_s[1], flat_s)
        elif not dmtree.is_nested(s):
            try:
                _, value = next(flat_sequence_it)
                return dmtree.MAP_TO_NONE if value is None else value
            except StopIteration:
                raise ValueError(
                    "Too few leaves provided by `flat_sequence` for "
                    f"`structure`. Got {len(flat_sequence)}."
                )
        return None

    ret = dmtree.traverse(unflatten_func, structure, top_down=True)
    try:
        index, _ = next(flat_sequence_it)
        raise ValueError(
            "Too many leaves provided by `flat_sequence` for `structure`; "
            f"expected: {index}, got {len(flat_sequence)}."
        )
    except StopIteration:
        return ret


def lists_to_tuples(structure):
    def list_to_tuple(instance):
        return tuple(instance) if isinstance(instance, list) else None

    return traverse(list_to_tuple, structure, top_down=False)


def map_shape_structure(func, structure):
    if not callable(func):
        raise TypeError(
            f"`func` must be callable, got {func} of type {type(func)}"
        )

    def map_shape_func(x):
        if isinstance(x, (list, tuple)) and all(
            isinstance(e, (int, type(None))) for e in x
        ):
            ret = func(x)
        elif is_nested(x):
            return None
        else:
            ret = func(x)
        return ret if ret is not None else dmtree.MAP_TO_NONE

    return traverse(map_shape_func, structure, top_down=True)
