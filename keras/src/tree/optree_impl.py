import optree
import optree.utils

from keras.src.backend.config import backend


def register_tree_node_class(cls):
    return optree.register_pytree_node_class(cls, namespace="keras")


# Register backend-specific node classes
if backend() == "tensorflow":
    from tensorflow.python.trackable.data_structures import ListWrapper
    from tensorflow.python.trackable.data_structures import _DictWrapper

    try:
        optree.register_pytree_node(
            ListWrapper,
            lambda x: (x, None),
            lambda metadata, children: ListWrapper(list(children)),
            namespace="keras",
        )

        def sorted_keys_and_values(d):
            keys = sorted(list(d.keys()))
            values = [d[k] for k in keys]
            return values, keys, keys

        optree.register_pytree_node(
            _DictWrapper,
            sorted_keys_and_values,
            lambda metadata, children: _DictWrapper(
                {key: child for key, child in zip(metadata, children)}
            ),
            namespace="keras",
        )
    except ValueError:
        pass  # We may have already registered if we are reiporting keras.


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
    # Detect MAP_TO_NONE without tree_api import to avoid circular import.
    if isinstance(ret, type) and ret.__name__ == "MAP_TO_NONE":
        return None
    return ret


def flatten(structure):
    # optree.tree_flatten returns a pair (leaves, treespec) where the first
    # element is a list of leaf values and the second element is a treespec
    # representing the structure of the pytree.
    leaves, _ = optree.tree_flatten(
        structure, none_is_leaf=True, namespace="keras"
    )
    return leaves


def flatten_with_path(structure):
    paths, leaves, _ = optree.tree_flatten_with_path(
        structure, none_is_leaf=True, namespace="keras"
    )
    return list(zip(paths, leaves))


def map_structure(func, *structures):
    if not structures:
        raise ValueError("Must provide at least one structure")

    # Add check for same structures, otherwise optree just maps to shallowest.
    def func_with_check(*args):
        if not all(
            optree.tree_is_leaf(s, none_is_leaf=True, namespace="keras")
            for s in args
        ):
            raise ValueError("Structures don't have the same nested structure.")
        return func(*args)

    map_func = func_with_check if len(structures) > 1 else func

    return optree.tree_map(
        map_func, *structures, none_is_leaf=True, namespace="keras"
    )


def map_structure_up_to(shallow_structure, func, *structures):
    if not structures:
        raise ValueError("Must provide at least one structure")

    # Add check that `shallow_structure` really is the shallowest.
    # Also only call `func` on `structures` and not `shallow_structure`.
    def func_with_check_without_shallow_structure(shallow, *args):
        if not optree.tree_is_leaf(shallow):
            raise ValueError("Structures don't have the same nested structure.")
        return func(*args)

    return optree.tree_map(
        func_with_check_without_shallow_structure,
        shallow_structure,
        *structures,
        none_is_leaf=True,
        namespace="keras",
    )


def assert_same_structure(a, b):
    def check(a_leaf, b_leaf):
        if not optree.tree_is_leaf(
            a_leaf, none_is_leaf=True, namespace="keras"
        ) or not optree.tree_is_leaf(
            b_leaf, none_is_leaf=True, namespace="keras"
        ):
            raise ValueError("Structures don't have the same nested structure.")
        return None

    optree.tree_map(check, a, b, none_is_leaf=True, namespace="keras")


def assert_same_paths(a, b):
    a_paths = set(optree.tree_paths(a, none_is_leaf=True, namespace="keras"))
    b_paths = set(optree.tree_paths(b, none_is_leaf=True, namespace="keras"))

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
    _, treespec = optree.tree_flatten(
        structure, none_is_leaf=True, namespace="keras"
    )
    return optree.tree_unflatten(treespec, flat_sequence)


def lists_to_tuples(structure):
    def list_to_tuple(instance):
        return tuple(instance) if isinstance(instance, list) else None

    return traverse(list_to_tuple, structure, top_down=False)


def map_shape_structure(func, structure):
    def is_shape_tuple(x):
        return isinstance(x, (list, tuple)) and all(
            isinstance(e, (int, type(None))) for e in x
        )

    return optree.tree_map(
        func,
        structure,
        is_leaf=is_shape_tuple,
        none_is_leaf=True,
        namespace="keras",
    )
