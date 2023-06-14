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
    """Produce random number based on the normal distribution.

    Args:
        shape: The shape of the random values to generate.
        mean: Floats, defaults to 0. Mean of the random values to generate.
        stddev: Floats, defaults to 1. Standard deviation of the random values
            to generate.
        dtype: Optional dtype of the tensor. Only floating point types are
            supported. If not specified, `keras.backend.floatx()` is used,
            which defaults to `float32` unless you configured it otherwise (via
            `keras.backend.set_floatx(float_dtype)`).
        seed: A Python integer or instance of
            `keras_core.backend.SeedGenerator`.
            Used to make the behavior of the initializer
            deterministic. Note that an initializer seeded with an integer
            or None (unseeded) will produce the same random values
            across multiple calls. To get different random values
            across multiple calls, use as seed an instance
            of `keras_core.backend.SeedGenerator`.
    """
    dtype = dtype or floatx()
    dtype = to_torch_dtype(dtype)
    generator = torch_seed_generator(seed)
    return torch.normal(
        mean, stddev, size=shape, generator=generator, dtype=dtype
    )


def uniform(shape, minval=0.0, maxval=1.0, dtype=None, seed=None):
    """Produce random number based on the uniform distribution.

    The generated values follow a uniform distribution in the range
    `[minval, maxval)`. The lower bound `minval` is included in the range,
    while the upper bound `maxval` is excluded.

    Args:
        shape: The shape of the random values to generate.
        minval: Floats, defaults to 0. Lower bound of the range of
            random values to generate (inclusive).
        maxval: Floats, defaults to 1. Upper bound of the range of
            random values to generate (exclusive).
        dtype: Optional dtype of the tensor. Only floating point types are
            supported. If not specified, `keras.backend.floatx()` is used,
            which defaults to `float32` unless you configured it otherwise (via
            `keras.backend.set_floatx(float_dtype)`)
        seed: A Python integer or instance of
            `keras_core.backend.SeedGenerator`.
            Used to make the behavior of the initializer
            deterministic. Note that an initializer seeded with an integer
            or None (unseeded) will produce the same random values
            across multiple calls. To get different random values
            across multiple calls, use as seed an instance
            of `keras_core.backend.SeedGenerator`.
    """
    dtype = dtype or floatx()
    dtype = to_torch_dtype(dtype)
    generator = torch_seed_generator(seed)
    return (maxval - minval) * torch.rand(
        *shape, generator=generator, dtype=dtype
    ) + minval


def truncated_normal(shape, mean=0.0, stddev=1.0, dtype=None, seed=None):
    """Produce random number based on the truncated normal distribution.

    The values are drawn from a normal distribution with specified mean and
    standard deviation, discarding and re-drawing any samples that are more
    than two standard deviations from the mean.

    Args:
        shape: The shape of the random values to generate.
        mean: Floats, defaults to 0. Mean of the random values to generate.
        stddev: Floats, defaults to 1. Standard deviation of the random values
            to generate.
        dtype: Optional dtype of the tensor. Only floating point types are
            supported. If not specified, `keras.backend.floatx()` is used,
            which defaults to `float32` unless you configured it otherwise (via
            `keras.backend.set_floatx(float_dtype)`)
        seed: A Python integer or instance of
            `keras_core.backend.SeedGenerator`.
            Used to make the behavior of the initializer
            deterministic. Note that an initializer seeded with an integer
            or None (unseeded) will produce the same random values
            across multiple calls. To get different random values
            across multiple calls, use as seed an instance
            of `keras_core.backend.SeedGenerator`.
    """
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
