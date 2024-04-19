import mlx.core as mx

from keras.src.backend.config import floatx
from keras.src.backend.mlx.core import to_mlx_dtype
from keras.src.random.seed_generator import SeedGenerator
from keras.src.random.seed_generator import draw_seed
from keras.src.random.seed_generator import make_default_seed


def mlx_draw_seed(seed):
    if isinstance(seed, mx.array):
        return seed
    else:
        return draw_seed(seed)


def normal(shape, mean=0.0, stddev=1.0, dtype=None, seed=None):
    dtype = dtype or floatx()
    dtype = to_mlx_dtype(dtype)
    seed = mlx_draw_seed(seed)
    sample = mx.random.normal(shape=shape, dtype=dtype, key=seed)
    return sample * stddev + mean


def uniform(shape, minval=0.0, maxval=1.0, dtype=None, seed=None):
    dtype = dtype or floatx()
    dtype = to_mlx_dtype(dtype)
    seed = mlx_draw_seed(seed)
    return mx.random.uniform(
        low=minval, high=maxval, shape=shape, dtype=dtype, key=seed
    )


def categorical(logits, num_samples, dtype="int32", seed=None):
    seed = mlx_draw_seed(seed)
    output = mx.random.categorical(logits, num_samples=num_samples, key=seed)
    return output.astype(to_mlx_dtype(dtype))


def randint(shape, minval, maxval, dtype="int32", seed=None):
    seed = mlx_draw_seed(seed)
    dtype = to_mlx_dtype(dtype)

    return mx.random.randint(
        low=minval, high=maxval, shape=shape, dtype=dtype, key=seed
    )


def truncated_normal(shape, mean=0.0, stddev=1.0, dtype=None, seed=None):
    dtype = dtype or floatx()
    dtype = to_mlx_dtype(dtype)
    seed = mlx_draw_seed(seed)
    sample = mx.random.truncated_normal(
        lower=-2.0, upper=2.0, shape=shape, dtype=dtype, key=seed
    )
    return sample * stddev + mean


def _get_concrete_noise_shape(inputs, noise_shape):
    if noise_shape is None:
        return inputs.shape

    concrete_inputs_shape = inputs.shape
    concrete_noise_shape = []
    for i, value in enumerate(noise_shape):
        concrete_noise_shape.append(
            concrete_inputs_shape[i] if value is None else value
        )
    return concrete_noise_shape


def dropout(inputs, rate, noise_shape=None, seed=None):
    seed = mlx_draw_seed(seed)
    keep_prob = 1.0 - rate
    # The `noise_shape` may contain `None` so we need to convert it
    # into a concrete shape before passing it on to jax.
    noise_shape = _get_concrete_noise_shape(inputs, noise_shape)
    mask = mx.random.bernoulli(p=keep_prob, shape=noise_shape, key=seed)
    mask = mx.broadcast_to(mask, inputs.shape)

    return mask * (inputs / keep_prob)


def shuffle(x, axis=0, seed=None):
    seed = mlx_draw_seed(seed)
    order = mx.argsort(mx.random.uniform(shape=(x.shape[axis],), key=seed))
    index = [slice(None)] * axis + [order]
    return x[index]


def gamma(shape, alpha, dtype=None, seed=None):
    raise NotImplementedError(
        "Sampling from Gamma distribution is not implemented in mlx"
    )


def binomial(shape, counts, probabilities, dtype=None, seed=None):
    raise NotImplementedError(
        "Sampling from a Binomial distribution is not implemented in mlx"
    )


def beta(shape, alpha, beta, dtype=None, seed=None):
    raise NotImplementedError(
        "Sampling from a Beta distribution is not implemented in mlx"
    )
