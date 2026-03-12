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


def _tree_is_leaf(tree, is_leaf=None):
    if is_leaf is not None and is_leaf(tree):
        return True
    return torch_tree._get_node_type(tree) not in torch_tree.SUPPORTED_NODES


def _dict_to_ordered_dict(structure):
    # We need to sort dict and defaultdict to ensure a deterministic order that
    # that is consistent with other tree implementations.
    def func(x):
        if type(x) is dict:
            return {k: x[k] for k in sorted(x.keys())}
        elif type(x) is defaultdict:
            return defaultdict(
                x.default_factory,
                {k: x[k] for k in sorted(x.keys())},
            )
        return None

    def traverse_children():
        children, treedef = torch_tree.tree_flatten(
            structure,
            is_leaf=lambda x: x is not structure,
        )
        if treedef.num_nodes == 1 and treedef.num_leaves == 1:
            return structure
        else:
            return torch_tree.tree_unflatten(
                [_dict_to_ordered_dict(c) for c in children],
                treedef,
            )

    ret = func(structure)
    if ret is None:
        return traverse_children()
    if isinstance(ret, type) and ret.__name__ == "MAP_TO_NONE":
        return None
    return ret


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

    structure = _dict_to_ordered_dict(structure)
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
    # We need to first sort dicts to ensure a deterministic order that is
    # consistent with other tree implementations.
    structure = _dict_to_ordered_dict(structure)
    leaves, _ = torch_tree.tree_flatten(structure)
    return leaves


def flatten_with_path(structure):
    # We need to first sort dicts to ensure a deterministic order that is
    # consistent with other tree implementations.
    structure = _dict_to_ordered_dict(structure)
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

    map_func = func
    if not none_is_leaf:

        def func_skipping_none(*args):
            # Check if the reference entry (first one) is None
            if args[0] is None:
                if not all(s is None for s in args):
                    raise ValueError(
                        "Structure mismatch: some arguments are None, others "
                        f"are not. Received arguments: {args}."
                    )
                return None
            return func(*args)

        map_func = func_skipping_none

    return torch_tree.tree_map(map_func, *structures)


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
    # We need to first sort dicts to ensure a deterministic order that is
    # consistent with other tree implementations.
    structure = _dict_to_ordered_dict(structure)
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

    # We need to first sort dicts to ensure a deterministic order that is
    # consistent with other tree implementations.
    structure = _dict_to_ordered_dict(structure)
    return torch_tree.tree_map(func, structure, is_leaf=is_shape_tuple)
