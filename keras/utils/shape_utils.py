def is_shape_tuple(x):
    if isinstance(x, (list, tuple)):
        if all(isinstance(e, (int, type(None))) for e in x):
            return True
    return False


def map_shape_structure(fn, struct):
    """Variant of tree.map_structure that operates on shape tuples."""
    if is_shape_tuple(struct):
        return fn(tuple(struct))
    if isinstance(struct, list):
        return [map_shape_structure(fn, e) for e in struct]
    if isinstance(struct, tuple):
        return tuple(map_shape_structure(fn, e) for e in struct)
    if isinstance(struct, dict):
        return {k: map_shape_structure(fn, v) for k, v in struct.items()}
    else:
        raise ValueError(f"Cannot map function to unknown object {struct}")
