import tensorflow as tf

from keras_core.backend.common.random import SeedGenerator
from keras_core.backend.common.random import draw_seed
from keras_core.backend.common.random import make_default_seed
from keras_core.backend.config import floatx


def tf_draw_seed(seed):
    # TF ops only accept int32/64 seeds but our base seed is uint32.
    return tf.cast(draw_seed(seed), dtype="int32")


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
    seed = tf_draw_seed(seed)
    return tf.random.stateless_normal(
        shape=shape, mean=mean, stddev=stddev, dtype=dtype, seed=seed
    )


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
    seed = tf_draw_seed(seed)
    return tf.random.stateless_uniform(
        shape=shape,
        minval=minval,
        maxval=maxval,
        dtype=dtype,
        seed=seed,
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
    seed = tf_draw_seed(seed)
    return tf.random.stateless_truncated_normal(
        shape=shape, mean=mean, stddev=stddev, dtype=dtype, seed=seed
    )


def dropout(inputs, rate, noise_shape=None, seed=None):
    seed = tf_draw_seed(seed)
    return tf.nn.experimental.stateless_dropout(
        inputs,
        rate=rate,
        noise_shape=noise_shape,
        seed=seed,
    )
