from keras.src.utils.module_utils import dmtree


def register_tree_node_class(cls):
    return cls


def is_nested(structure):
    return dmtree.is_nested(structure)


def traverse(func, structure, top_down=True):
    return dmtree.traverse(func, structure, top_down=top_down)


def flatten(structure):
    return dmtree.flatten(structure)


def flatten_with_path(structure):
    return dmtree.flatten_with_path(structure)


def map_structure(func, *structures):
    return dmtree.map_structure(func, *structures)


def map_structure_up_to(shallow_structure, func, *structures):
    return dmtree.map_structure_up_to(shallow_structure, func, *structures)


def assert_same_structure(a, b, check_types=True):
    return dmtree.assert_same_structure(a, b, check_types=check_types)


def assert_same_paths(a, b):
    a_paths = set([path for path, leaf in dmtree.flatten_with_path(a)])
    b_paths = set([path for path, leaf in dmtree.flatten_with_path(b)])

    if a_paths != b_paths:
        msg = "`a` and `b` don't have the same paths."
        a_diff = a_paths.difference(b_paths)
        if a_diff:
            msg += f"\nPaths in `a` missing in `b`:\n{a_diff}"
        b_diff = b_paths.difference(a_paths)
        if b_diff:
            msg += f"\nPaths in `b` missing in `a`:\n{b_diff}"
        raise ValueError(msg)


def pack_sequence_as(structure, flat_sequence, sequence_fn=None):
    is_nested_fn = dmtree.is_nested
    sequence_fn = sequence_fn or dmtree._sequence_like

    def truncate(value, length):
        value_str = str(value)
        return value_str[:length] + (value_str[length:] and "...")

    if not is_nested_fn(flat_sequence):
        raise TypeError(
            "Attempted to pack value:\n  {}\ninto a structure, but found "
            "incompatible type `{}` instead.".format(
                truncate(flat_sequence, 100), type(flat_sequence)
            )
        )

    if not is_nested_fn(structure):
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

    packed = []
    try:
        final_index, packed = packed_nest_with_indices(
            structure, flat_sequence, 0, is_nested_fn, sequence_fn
        )
        if final_index < len(flat_sequence):
            raise IndexError
    except IndexError:
        flat_structure = dmtree.flatten(structure)
        if len(flat_structure) != len(flat_sequence):
            # pylint: disable=raise-missing-from
            raise ValueError(
                "Could not pack sequence. "
                f"Structure had {len(flat_structure)} atoms, but "
                f"flat_sequence had {len(flat_sequence)} items. "
                f"Structure: {structure}, flat_sequence: {flat_sequence}."
            )
    return sequence_fn(structure, packed)


def packed_nest_with_indices(
    structure, flat, index, is_nested_fn, sequence_fn=None
):
    """Helper function for pack_sequence_as.

    Args:
        structure: structure to mimic.
        flat: Flattened values to output substructure for.
        index: Index at which to start reading from flat.
        is_nested_fn: Function used to test if a value should
            be treated as a nested structure.
        sequence_fn: Function used to generate a new structure instance.

    Returns:
        The tuple (new_index, child), where:
        * new_index - the updated index into `flat`
            having processed `structure`.
        * packed - the subset of `flat` corresponding to `structure`,
            having started at `index`, and packed into the same nested
            format.
    """
    packed = []
    sequence_fn = sequence_fn or dmtree._sequence_like
    for s in yield_value(structure):
        if is_nested_fn(s):
            new_index, child = packed_nest_with_indices(
                s, flat, index, is_nested_fn, sequence_fn
            )
            packed.append(sequence_fn(s, child))
            index = new_index
        else:
            packed.append(flat[index])
            index += 1
    return index, packed


def yield_value(iterable):
    for _, v in dmtree._yield_sorted_items(iterable):
        yield v


def lists_to_tuples(structure):
    def sequence_fn(instance, args):
        if isinstance(instance, list):
            return tuple(args)
        return dmtree._sequence_like(instance, args)

    return pack_sequence_as(
        structure,
        dmtree.flatten(structure),
        sequence_fn=sequence_fn,
    )


def is_shape_tuple(x):
    if isinstance(x, (list, tuple)):
        if all(isinstance(e, (int, type(None))) for e in x):
            return True
    return False


def map_shape_structure(func, structure):
    if is_shape_tuple(structure):
        return func(tuple(structure))
    if isinstance(structure, list):
        return [map_shape_structure(func, e) for e in structure]
    if isinstance(structure, tuple):
        return tuple(map_shape_structure(func, e) for e in structure)
    if isinstance(structure, dict):
        return {k: map_shape_structure(func, v) for k, v in structure.items()}
    else:
        raise ValueError(f"Cannot map function to unknown object {structure}")
