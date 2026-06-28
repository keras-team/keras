import math

import mlx.core as mx
import numpy as np

from keras.src.backend.config import floatx
from keras.src.backend.mlx.core import _mlx_dtype
from keras.src.backend.mlx.core import convert_to_tensor
from keras.src.random.seed_generator import SeedGenerator  # noqa: F401
from keras.src.random.seed_generator import draw_seed
from keras.src.random.seed_generator import make_default_seed  # noqa: F401


def _key(seed):
    # `draw_seed` returns a uint32[2] array, which is exactly an MLX PRNG key.
    return draw_seed(seed)


def normal(shape, mean=0.0, stddev=1.0, dtype=None, seed=None):
    dtype = dtype or floatx()
    return mx.random.normal(
        shape=tuple(shape),
        loc=mean,
        scale=stddev,
        key=_key(seed),
        dtype=_mlx_dtype(dtype),
    )


def uniform(shape, minval=0.0, maxval=1.0, dtype=None, seed=None):
    dtype = dtype or floatx()
    return mx.random.uniform(
        low=minval,
        high=maxval,
        shape=tuple(shape),
        key=_key(seed),
        dtype=_mlx_dtype(dtype),
    )


def categorical(logits, num_samples, dtype="int64", seed=None):
    # `logits` shape: (batch_size, num_classes); sample `num_samples` per row.
    # `mx.random.categorical` requires an MLX array and natively supports drawing
    # `num_samples` per row in one call (output shape (batch, num_samples)).
    logits = convert_to_tensor(logits)
    out = mx.random.categorical(logits, num_samples=num_samples, key=_key(seed))
    return out.astype(_mlx_dtype(dtype))


def randint(shape, minval, maxval, dtype="int32", seed=None):
    # MLX randint is half-open [low, high).
    if maxval is not None:
        high = maxval
    else:
        high = minval + 1
    low = minval
    return mx.random.randint(
        low=low,
        high=high,
        shape=tuple(shape),
        key=_key(seed),
        dtype=_mlx_dtype(dtype),
    )


def truncated_normal(shape, mean=0.0, stddev=1.0, dtype=None, seed=None):
    dtype = dtype or floatx()
    lower = mean - 2 * stddev
    upper = mean + 2 * stddev
    return mx.random.truncated_normal(
        lower=lower,
        upper=upper,
        shape=tuple(shape),
        key=_key(seed),
        dtype=_mlx_dtype(dtype),
    )


def dropout(inputs, rate, noise_shape=None, seed=None):
    if rate == 1.0:
        return mx.zeros_like(inputs)
    if rate == 0.0:
        return inputs
    dtype = inputs.dtype
    key = _key(seed)

    keep_prob = 1.0 - rate
    if noise_shape is None:
        noise_shape = inputs.shape
    else:
        noise_shape = [
            n if n is not None else inputs.shape[i]
            for i, n in enumerate(noise_shape)
        ]

    mask = mx.random.bernoulli(
        mx.array(keep_prob), shape=tuple(noise_shape)
    )
    mask = mx.broadcast_to(mask, inputs.shape)
    return mx.where(
        mask,
        (inputs / keep_prob).astype(dtype),
        mx.zeros_like(inputs),
    )


def shuffle(x, axis=0, seed=None):
    x = convert_to_tensor(x)
    return mx.random.permutation(x, axis=axis, key=_key(seed))


def gamma(shape, alpha, dtype=None, seed=None):
    # No native MLX gamma; generate via NumPy and convert.
    dtype = dtype or floatx()
    seed = draw_seed(seed)
    rng = np.random.default_rng(np.asarray(seed))
    return convert_to_tensor(
        rng.gamma(alpha, scale=1.0, size=shape), dtype=dtype
    )


def binomial(shape, counts, probabilities, dtype=None, seed=None):
    dtype = dtype or floatx()
    seed = draw_seed(seed)
    rng = np.random.default_rng(np.asarray(seed))
    sample = rng.binomial(n=counts, p=probabilities, size=shape)
    return convert_to_tensor(sample, dtype=dtype)


def beta(shape, alpha, beta, dtype=None, seed=None):
    dtype = dtype or floatx()
    seed = draw_seed(seed)
    rng = np.random.default_rng(np.asarray(seed))
    sample = rng.beta(a=alpha, b=beta, size=shape)
    return convert_to_tensor(sample, dtype=dtype)
