import jax

from keras_core.backend.config import floatx
from keras_core.random.seed_generator import SeedGenerator
from keras_core.random.seed_generator import draw_seed
from keras_core.random.seed_generator import make_default_seed


def normal(shape, mean=0.0, stddev=1.0, dtype=None, seed=None):
    """Draw random samples from a normal (Gaussian) distribution.

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
    seed = draw_seed(seed)
    sample = jax.random.normal(seed, shape=shape, dtype=dtype)
    return sample * stddev + mean


def uniform(shape, minval=0.0, maxval=1.0, dtype=None, seed=None):
    """Draw samples from a uniform distribution.

    The generated values follow a uniform distribution in the range
    `[minval, maxval)`. The lower bound `minval` is included in the range,
    while the upper bound `maxval` is excluded.

    For floats, the default range is `[0, 1)`.  For ints, at least `maxval`
    must be specified explicitly.

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
    seed = draw_seed(seed)
    return jax.random.uniform(
        seed, shape=shape, dtype=dtype, minval=minval, maxval=maxval
    )


def truncated_normal(shape, mean=0.0, stddev=1.0, dtype=None, seed=None):
    """Draw samples from a truncated normal distribution.

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
    dtype = dtype or floatx()
    seed = draw_seed(seed)
    sample = jax.random.truncated_normal(
        seed, shape=shape, lower=-2.0, upper=2.0, dtype=dtype
    )
    return sample * stddev + mean


def _get_concrete_noise_shape(inputs, noise_shape):
    if noise_shape is None:
        return inputs.shape

    concrete_inputs_shape = inputs.shape
    noise_shape = []
    for i, value in enumerate(noise_shape):
        noise_shape.append(concrete_inputs_shape[i] if value is None else value)
    return noise_shape


def dropout(inputs, rate, noise_shape=None, seed=None):
    seed = draw_seed(seed)
    keep_prob = 1.0 - rate
    # The `noise_shape` may contain `None` so we need to convert it
    # into a concrete shape before passing it on to jax.
    noise_shape = _get_concrete_noise_shape(inputs, noise_shape)
    mask = jax.random.bernoulli(seed, p=keep_prob, shape=noise_shape)
    mask = jax.numpy.broadcast_to(mask, inputs.shape)
    return jax.lax.select(
        mask, inputs / keep_prob, jax.numpy.zeros_like(inputs)
    )
