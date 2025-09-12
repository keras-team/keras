from collections import defaultdict

from torch.utils import _pytree as torch_tree


def register_tree_node_class(cls):
    torch_tree.register_pytree_node(
        cls,
        flatten_fn=lambda x: x.torchtree_flatten(),
        unflatten_fn=cls.torchtree_unflatten,
        serialized_type_name=f"{cls.__name__}",
        flatten_with_keys_fn=lambda x: x.torchtree_flatten_with_keys(),
    )
    return cls


# Re-register dict and defaultdict nodes to ensure the consistent behavior
# compared to optree.
def _deregister_pytree_node(cls):
    with torch_tree._NODE_REGISTRY_LOCK:
        del torch_tree.SUPPORTED_NODES[cls]
        node_def = torch_tree.SUPPORTED_SERIALIZED_TYPES[cls]
        del torch_tree.SERIALIZED_TYPE_TO_PYTHON_TYPE[
            node_def.serialized_type_name
        ]
        del torch_tree.SUPPORTED_SERIALIZED_TYPES[cls]
        if hasattr(torch_tree, "CONSTANT_NODES"):
            torch_tree.CONSTANT_NODES.discard(cls)


_deregister_pytree_node(dict)
_deregister_pytree_node(defaultdict)


def _dict_flatten(d):
    keys = sorted(d.keys())
    values = [d[k] for k in keys]
    return values, keys


def _dict_flatten_with_keys(d):
    values, context = _dict_flatten(d)
    return [
        (torch_tree.MappingKey(k), v) for k, v in zip(context, values)
    ], context


def _defaultdict_flatten(d):
    values, dict_context = _dict_flatten(d)
    return values, [d.default_factory, dict_context]


def _defaultdict_flatten_with_keys(d):
    values, context = _defaultdict_flatten(d)
    _, dict_context = context
    return [
        (torch_tree.MappingKey(k), v) for k, v in zip(dict_context, values)
    ], context


torch_tree._private_register_pytree_node(
    dict,
    _dict_flatten,
    torch_tree._dict_unflatten,
    serialized_type_name="builtins.dict",
    flatten_with_keys_fn=_dict_flatten_with_keys,
)
torch_tree._private_register_pytree_node(
    defaultdict,
    _defaultdict_flatten,
    torch_tree._defaultdict_unflatten,
    serialized_type_name="collections.defaultdict",
    to_dumpable_context=torch_tree._defaultdict_serialize,
    from_dumpable_context=torch_tree._defaultdict_deserialize,
    flatten_with_keys_fn=_defaultdict_flatten_with_keys,
)


def _tree_is_leaf(tree):
    return torch_tree._get_node_type(tree) not in torch_tree.SUPPORTED_NODES


def is_nested(structure):
    return not _tree_is_leaf(structure)


def traverse(func, structure, top_down=True):
    def traverse_children():
        children, treedef = torch_tree.tree_flatten(
            structure,
            is_leaf=lambda x: x is not structure,
        )
        if treedef.num_nodes == 1 and treedef.num_leaves == 1:
            return structure
        else:
            return torch_tree.tree_unflatten(
                [traverse(func, c, top_down=top_down) for c in children],
                treedef,
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
    # torch_tree.tree_flatten returns a pair (leaves, treespec) where the first
    # element is a list of leaf values and the second element is a treespec
    # representing the structure of the pytree.
    leaves, _ = torch_tree.tree_flatten(structure)
    return leaves


def flatten_with_path(structure):
    leaves_with_path, _ = torch_tree.tree_flatten_with_path(structure)
    results = []
    fields = []
    for key, leaf in leaves_with_path:
        for k in key:
            if isinstance(k, torch_tree.GetAttrKey) and k.name not in fields:
                fields.append(k.name)
    fields = sorted(fields)
    field_to_idx = {f: i for i, f in enumerate(fields)}
    for key, leaf in leaves_with_path:
        # Convert to a tuple of keys.
        path = []
        for k in key:
            if isinstance(k, torch_tree.SequenceKey):
                path.append(k.idx)
            elif isinstance(k, torch_tree.MappingKey):
                path.append(k.key)
            elif isinstance(k, torch_tree.GetAttrKey):
                path.append(field_to_idx[k.name])
        results.append((tuple(path), leaf))
    return results


def map_structure(func, *structures, none_is_leaf=True):
    if not structures:
        raise ValueError("Must provide at least one structure")

    def tree_is_leaf(x, none_is_leaf=True):
        if none_is_leaf:
            return _tree_is_leaf(x) or x is None
        else:
            return _tree_is_leaf(x)

    # Add check for same structures, otherwise torch_tree just maps to
    # shallowest.
    def func_with_check(*args):
        if not all(tree_is_leaf(s, none_is_leaf=none_is_leaf) for s in args):
            raise ValueError("Structures don't have the same nested structure.")
        return func(*args)

    map_func = func_with_check if len(structures) > 1 else func

    return torch_tree.tree_map(
        map_func,
        *structures,
        is_leaf=lambda x: tree_is_leaf(x, none_is_leaf=none_is_leaf),
    )


def map_structure_up_to(shallow_structure, func, *structures):
    if not structures:
        raise ValueError("Must provide at least one structure")

    # Add check that `shallow_structure` really is the shallowest.
    # Also only call `func` on `structures` and not `shallow_structure`.
    def func_with_check_without_shallow_structure(shallow, *args):
        if not _tree_is_leaf(shallow):
            raise ValueError("Structures don't have the same nested structure.")
        return func(*args)

    return torch_tree.tree_map(
        func_with_check_without_shallow_structure,
        shallow_structure,
        *structures,
    )


def assert_same_structure(a, b):
    def check(a_leaf, b_leaf):
        if not _tree_is_leaf(a_leaf) or not _tree_is_leaf(b_leaf):
            raise ValueError("Structures don't have the same nested structure.")
        return None

    torch_tree.tree_map(check, a, b)


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
    _, treespec = torch_tree.tree_flatten(structure)
    return torch_tree.tree_unflatten(flat_sequence, treespec)


def lists_to_tuples(structure):
    def list_to_tuple(instance):
        return tuple(instance) if isinstance(instance, list) else None

    return traverse(list_to_tuple, structure, top_down=False)


def map_shape_structure(func, structure):
    def is_shape_tuple(x):
        return isinstance(x, (list, tuple)) and all(
            isinstance(e, (int, type(None))) for e in x
        )

    return torch_tree.tree_map(func, structure, is_leaf=is_shape_tuple)
