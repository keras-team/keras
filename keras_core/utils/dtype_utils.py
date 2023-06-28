from keras_core import backend
from keras_core import ops


def dtype_size(dtype):
    if dtype in ("bfloat16", "float16"):
        return 16
    if dtype in ("float32", "int32"):
        return 32
    if dtype in ("float64", "int64"):
        return 64
    if dtype == "uint8":
        return 8
    if dtype == "bool":
        return 1
    raise ValueError(f"Invalid dtype: {dtype}")


def is_float(dtype):
    return "float" in dtype


def cast_to_common_dtype(tensors):
    """Cast a list of tensors to a common dtype.

    If any tensor is floating-point, they will all be casted to the most-precise
    floating-point dtype. Otherwise the tensors are not casted.

    Args:
        tensors: A list of tensors.

    Returns:
        Same list, casted to a common dtype.
    """
    highest_float = None
    for x in tensors:
        dtype = backend.standardize_dtype(x.dtype)
        if is_float(dtype):
            if highest_float is None or dtype_size(dtype) > highest_float:
                highest_float = dtype
            elif dtype == "float16" and highest_float == "bfloat16":
                highest_float = "float32"
    if highest_float:
        tensors = [ops.cast(x, highest_float) for x in tensors]
    return tensors
