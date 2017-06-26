import numpy as np
import torch

from ..common import floatx
from .shape import cast


def random_normal(shape, mean=0.0, stddev=1.0, dtype=None, seed=None):
    if dtype is None:
        dtype = floatx()
    if seed is not None:
        torch.manual_seed(seed)
    x = torch.randn(shape)
    x *= stddev
    x += mean
    return cast(x, dtype)


def random_uniform(shape, minval=0.0, maxval=1.0, dtype=None, seed=None):
    if dtype is None:
        dtype = floatx()
    if seed is not None:
        torch.manual_seed(seed)
    x = torch.rand(shape)
    x *= (maxval - minval)
    x += minval
    return cast(x, dtype)


def random_binomial(shape, p=0.0, dtype=None, seed=None):
    if dtype is None:
        dtype = floatx()
    if seed is not None:
        torch.manual_seed(seed)
    if p <= 0.0:
        x = torch.zeros(shape)
    elif 1.0 <= p:
        x = torch.ones(shape)
    else:
        x = torch.rand(shape)
        x /= p
        x.floor_()
        x = 1 - x
        x.clamp_(0, 1)
    return cast(x, dtype)


def truncated_normal(shape, mean=0.0, stddev=1.0, dtype=None, seed=None):
    if dtype is None:
        dtype = floatx()
    if seed is not None:
        torch.manual_seed(seed)
    x = torch.randn(shape)
    x.clamp_(-2, 2)
    x *= stddev
    x += mean
    return cast(x, dtype)
