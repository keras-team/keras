import torch

from keras.backend.torch.core import convert_to_tensor


def cholesky(x):
    x = convert_to_tensor(x)
    return torch.cholesky(x)


def det(x):
    x = convert_to_tensor(x)
    return torch.det(x)


def inv(x):
    x = convert_to_tensor(x)
    return torch.linalg.inv(x)


def solve(a, b):
    a = convert_to_tensor(a)
    b = convert_to_tensor(b)
    return torch.linalg.solve(a, b)


def solve_triangular(a, b, lower=False):
    a = convert_to_tensor(a)
    b = convert_to_tensor(b)
    return torch.linalg.solve_triangular(a, b, upper=~lower)
