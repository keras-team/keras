import torch
import torch.nn.functional as tnn

from keras_core.backend.config import floatx
from keras_core.backend.torch.core import to_torch_dtype
from keras_core.random.seed_generator import SeedGenerator
from keras_core.random.seed_generator import draw_seed
from keras_core.random.seed_generator import make_default_seed


def torch_seed_generator(seed):
    seed_val, _ = draw_seed(seed)
    generator = torch.Generator()
    generator.manual_seed(int(seed_val))
    return generator


def normal(shape, mean=0.0, stddev=1.0, dtype=None, seed=None):
    dtype = dtype or floatx()
    dtype = to_torch_dtype(dtype)
    generator = torch_seed_generator(seed)
    return torch.normal(
        mean, stddev, size=shape, generator=generator, dtype=dtype
    )


def uniform(shape, minval=0.0, maxval=1.0, dtype=None, seed=None):
    dtype = dtype or floatx()
    dtype = to_torch_dtype(dtype)
    generator = torch_seed_generator(seed)
    return (maxval - minval) * torch.rand(
        *shape, generator=generator, dtype=dtype
    ) + minval


def randint(shape, minval, maxval, dtype="int32", seed=None):
    dtype = to_torch_dtype(dtype)
    generator = torch_seed_generator(seed)
    return torch.randint(
        low=minval, high=maxval, size=shape, generator=generator, dtype=dtype
    )


def truncated_normal(shape, mean=0.0, stddev=1.0, dtype=None, seed=None):
    # Take a larger standard normal dist, discard values outside 2 * stddev
    # Offset by mean and stddev
    x = normal(shape + (4,), mean=0, stddev=1, dtype=dtype, seed=seed)
    valid = (x > -2) & (x < 2)
    indexes = valid.max(-1, keepdim=True)[1]
    trunc_x = torch.empty(shape)
    trunc_x.data.copy_(x.gather(-1, indexes).squeeze(-1))
    trunc_x.data.mul_(stddev).add_(mean)
    return trunc_x


def dropout(inputs, rate, noise_shape=None, seed=None):
    # TODO: setting seed globally via `manual_seed` might create side effects.
    if seed is not None:
        seed_val, _ = draw_seed(seed)
        torch.manual_seed(int(seed_val))
    return tnn.dropout(inputs, p=rate)
