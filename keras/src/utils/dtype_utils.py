from keras.src import backend
from keras.src import ops

DTYPE_TO_SIZE = {
    **{f"float{i}": i for i in (16, 32, 64)},
    **{f"int{i}": i for i in (8, 16, 32, 64)},
    **{f"uint{i}": i for i in (8, 16, 32, 64)},
    "bfloat16": 16,
    "bool": 1,
}


def dtype_size(dtype):
    size = DTYPE_TO_SIZE.get(dtype, None)
    if size is None:
        raise ValueError(f"Invalid dtype: {dtype}")
    return size


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
    highest_float_size = (
        -1
    )  # Initially set to an impossible value for comparison
    for x in tensors:
        dtype = backend.standardize_dtype(x.dtype)
        if is_float(dtype):
            if highest_float is None or dtype_size(dtype) > highest_float_size:
                highest_float = dtype
                highest_float_size = dtype_size(dtype)
            elif dtype == "float16" and highest_float == "bfloat16":
                highest_float = "float32"
                highest_float_size = dtype_size(highest_float)
    if highest_float:
        tensors = [ops.cast(x, highest_float) for x in tensors]
    return tensors
