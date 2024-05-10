import mlx.core as mx

from keras.src.backend.common import standardize_dtype
from keras.src.backend.mlx.core import convert_to_tensor


def norm(x, ord=None, axis=None, keepdims=False):
    dtype = standardize_dtype(x.dtype)
    if "int" in dtype or dtype == "bool":
        dtype = dtypes.result_type(x.dtype, "float32")
    x = convert_to_tensor(x, dtype=dtype)
    return mx.linalg.norm(x, ord=ord, axis=axis, keepdims=keepdims)
