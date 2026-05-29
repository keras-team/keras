import numpy as np
import paddle

from keras.src.backend.config import floatx
from keras.src.backend.paddle.core import PADDLE_DTYPES
from keras.src.backend.paddle.core import convert_to_numpy
from keras.src.backend.paddle.core import convert_to_tensor
from keras.src.backend.paddle.core import to_paddle_dtype
from keras.src.random.seed_generator import SeedGenerator
from keras.src.random.seed_generator import draw_seed
from keras.src.random.seed_generator import make_default_seed


def normal(shape, mean=0.0, stddev=1.0, dtype=None, seed=None):
    dtype = dtype or floatx()
    paddle_dtype = to_paddle_dtype(dtype)
    seed = draw_seed(seed)
    if isinstance(seed, paddle.Tensor):
        seed = convert_to_numpy(seed)
    rng = np.random.default_rng(seed)
    data = rng.normal(size=shape, loc=mean, scale=stddev).astype(dtype)
    return paddle.to_tensor(data, dtype=paddle_dtype)


def uniform(shape, minval=0.0, maxval=1.0, dtype=None, seed=None):
    dtype = dtype or floatx()
    paddle_dtype = to_paddle_dtype(dtype)
    seed = draw_seed(seed)
    if isinstance(seed, paddle.Tensor):
        seed = convert_to_numpy(seed)
    rng = np.random.default_rng(seed)
    data = rng.uniform(size=shape, low=minval, high=maxval).astype(dtype)
    return paddle.to_tensor(data, dtype=paddle_dtype)


def categorical(logits, num_samples, dtype="int64", seed=None):
    raise NotImplementedError(
        "`categorical` is not supported with paddle backend"
    )


def randint(shape, minval, maxval, dtype="int32", seed=None):
    raise NotImplementedError(
        "`randint` is not supported with paddle backend"
    )


def truncated_normal(shape, mean=0.0, stddev=1.0, dtype=None, seed=None):
    dtype = dtype or floatx()
    paddle_dtype = to_paddle_dtype(dtype)
    seed = draw_seed(seed)
    if isinstance(seed, paddle.Tensor):
        seed = convert_to_numpy(seed)
    rng = np.random.default_rng(seed)

    lower_bound = mean - 2 * stddev
    upper_bound = mean + 2 * stddev

    flat_shape = np.prod(shape)
    random_numbers = np.empty(0)

    while random_numbers.shape[0] < flat_shape:
        batch = rng.normal(loc=mean, scale=stddev, size=flat_shape)
        valid = batch[(batch >= lower_bound) & (batch <= upper_bound)]
        random_numbers = np.append(random_numbers, valid)

    np_array_res = random_numbers[:flat_shape].astype(dtype).reshape(shape)
    return paddle.to_tensor(np_array_res, dtype=paddle_dtype)


def dropout(inputs, rate, noise_shape=None, seed=None):
    raise NotImplementedError(
        "`dropout` is not supported with paddle backend"
    )


def shuffle(x, axis=0):
    raise NotImplementedError(
        "`shuffle` is not supported with paddle backend"
    )


def gamma(shape, alpha, dtype=None, seed=None):
    raise NotImplementedError(
        "`gamma` is not supported with paddle backend"
    )


def binomial(shape, counts, probabilities, dtype=None, seed=None):
    raise NotImplementedError(
        "`binomial` is not supported with paddle backend"
    )


def beta(shape, alpha, beta, dtype=None, seed=None):
    raise NotImplementedError(
        "`beta` is not supported with paddle backend"
    )


def seed_generator():
    return SeedGenerator(seed=make_default_seed())
