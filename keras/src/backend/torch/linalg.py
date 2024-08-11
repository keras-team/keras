import torch

from keras.src.backend import config
from keras.src.backend import standardize_dtype
from keras.src.backend.common import dtypes
from keras.src.backend.torch.core import cast
from keras.src.backend.torch.core import convert_to_tensor


def cholesky(x):
    return torch.linalg.cholesky(x)


def det(x):
    return torch.det(x)


def eig(x):
    return torch.linalg.eig(x)


def eigh(x):
    return torch.linalg.eigh(x)


def inv(x):
    return torch.linalg.inv(x)


def lu_factor(x):
    LU, pivots = torch.linalg.lu_factor(x)
    # torch retuns pivots with 1-based indexing
    return LU, pivots - 1


def norm(x, ord=None, axis=None, keepdims=False):
    x = convert_to_tensor(x)
    if standardize_dtype(x.dtype) == "int64":
        dtype = config.floatx()
    else:
        dtype = dtypes.result_type(x.dtype, float)
    x = cast(x, dtype)
    return torch.linalg.norm(x, ord=ord, dim=axis, keepdim=keepdims)


def qr(x, mode="reduced"):
    if mode not in {"reduced", "complete"}:
        raise ValueError(
            "`mode` argument value not supported. "
            "Expected one of {'reduced', 'complete'}. "
            f"Received: mode={mode}"
        )
    return torch.linalg.qr(x, mode=mode)


def solve(a, b):
    return torch.linalg.solve(a, b)


def solve_triangular(a, b, lower=False):
    if b.ndim == a.ndim - 1:
        b = torch.unsqueeze(b, axis=-1)
        return torch.linalg.solve_triangular(a, b, upper=not lower).squeeze(
            axis=-1
        )
    return torch.linalg.solve_triangular(a, b, upper=not lower)


def svd(x, full_matrices=True, compute_uv=True):
    if not compute_uv:
        raise NotImplementedError(
            "`compute_uv=False` is not supported for torch backend."
        )
    return torch.linalg.svd(x, full_matrices=full_matrices)


def lstsq(a, b, rcond=None):
    a = convert_to_tensor(a)
    b = convert_to_tensor(b)
    return torch.linalg.lstsq(a, b, rcond=rcond)[0]
