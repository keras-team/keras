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
    if seed is not None:
        seed_val = draw_seed(seed)
        if isinstance(seed_val, paddle.Tensor):
            seed_val = int(convert_to_numpy(seed_val).flat[0])
        paddle.seed(seed_val)
    return paddle.normal(mean=mean, std=stddev, shape=shape).cast(paddle_dtype)


def uniform(shape, minval=0.0, maxval=1.0, dtype=None, seed=None):
    dtype = dtype or floatx()
    paddle_dtype = to_paddle_dtype(dtype)
    if seed is not None:
        seed_val = draw_seed(seed)
        if isinstance(seed_val, paddle.Tensor):
            seed_val = int(convert_to_numpy(seed_val).flat[0])
        paddle.seed(seed_val)
    return paddle.uniform(shape=shape, min=minval, max=maxval).cast(
        paddle_dtype
    )


def categorical(logits, num_samples, dtype="int64", seed=None):
    logits = convert_to_tensor(logits)
    if seed is not None:
        seed_val = draw_seed(seed)
        if isinstance(seed_val, paddle.Tensor):
            seed_val = int(convert_to_numpy(seed_val).flat[0])
        paddle.seed(seed_val)
    probs = paddle.nn.functional.softmax(logits, axis=-1)
    return paddle.multinomial(probs, num_samples=num_samples).cast(
        to_paddle_dtype(dtype)
    )


def randint(shape, minval, maxval, dtype="int32", seed=None):
    if seed is not None:
        seed_val = draw_seed(seed)
        if isinstance(seed_val, paddle.Tensor):
            seed_val = int(convert_to_numpy(seed_val).flat[0])
        paddle.seed(seed_val)
    paddle_dtype = to_paddle_dtype(dtype)
    return paddle.randint(minval, maxval, shape=shape, dtype=paddle_dtype)


def truncated_normal(shape, mean=0.0, stddev=1.0, dtype=None, seed=None):
    dtype = dtype or floatx()
    paddle_dtype = to_paddle_dtype(dtype)
    if seed is not None:
        seed_val = draw_seed(seed)
        if isinstance(seed_val, paddle.Tensor):
            seed_val = int(convert_to_numpy(seed_val).flat[0])
        paddle.seed(seed_val)
    from scipy.stats import truncnorm

    a, b = -2.0, 2.0
    values = truncnorm.rvs(a, b, loc=mean, scale=stddev, size=shape)
    return paddle.to_tensor(values, dtype=paddle_dtype)


def dropout(inputs, rate, noise_shape=None, seed=None):
    if rate == 0:
        return inputs
    inputs = convert_to_tensor(inputs)
    if noise_shape is None:
        noise_shape = paddle.shape(inputs)
    if seed is not None:
        seed_val = draw_seed(seed)
        if isinstance(seed_val, paddle.Tensor):
            seed_val = int(convert_to_numpy(seed_val).flat[0])
        paddle.seed(seed_val)
    if rate == 1.0:
        return paddle.zeros_like(inputs)
    keep_mask = paddle.bernoulli(
        paddle.full(noise_shape, 1.0 - rate, dtype="float32")
    )
    return inputs * keep_mask / (1.0 - rate)


def shuffle(x, axis=0):
    x = convert_to_tensor(x)
    indices = paddle.randperm(x.shape[axis])
    return paddle.index_select(x, indices, axis=axis)


def gamma(shape, alpha, dtype=None, seed=None):
    raise NotImplementedError("`gamma` is not supported with paddle backend")


def binomial(shape, counts, probabilities, dtype=None, seed=None):
    raise NotImplementedError("`binomial` is not supported with paddle backend")


def beta(shape, alpha, beta, dtype=None, seed=None):
    raise NotImplementedError("`beta` is not supported with paddle backend")


def seed_generator():
    return SeedGenerator(seed=make_default_seed())
