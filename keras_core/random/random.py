from keras_core import backend
from keras_core.api_export import keras_core_export


@keras_core_export("keras_core.random.normal")
def normal(shape, mean=0.0, stddev=1.0, dtype=None, seed=None):
    """Draw random samples from a normal (Gaussian) distribution.

    Args:
        shape: The shape of the random values to generate.
        mean: Floats, defaults to 0. Mean of the random values to generate.
        stddev: Floats, defaults to 1. Standard deviation of the random values
            to generate.
        dtype: Optional dtype of the tensor. Only floating point types are
            supported. If not specified, `keras_core.config.floatx()` is used,
            which defaults to `float32` unless you configured it otherwise (via
            `keras_core.config.set_floatx(float_dtype)`).
        seed: A Python integer or instance of
            `keras_core.random.SeedGenerator`.
            Used to make the behavior of the initializer
            deterministic. Note that an initializer seeded with an integer
            or None (unseeded) will produce the same random values
            across multiple calls. To get different random values
            across multiple calls, use as seed an instance
            of `keras_core.random.SeedGenerator`.
    """
    return backend.random.normal(
        shape, mean=mean, stddev=stddev, dtype=dtype, seed=seed
    )


@keras_core_export("keras_core.random.categorical")
def categorical(logits, num_samples, dtype="int32", seed=None):
    """Draws samples from a categorical distribution.

    This function takes as input `logits`, a 2-D input tensor with shape
    (batch_size, num_classes). Each row of the input represents a categorical
    distribution, with each column index containing the log-probability for a
    given class.

    The function will output a 2-D tensor with shape (batch_size, num_samples),
    where each row contains samples from the corresponding row in `logits`.
    Each column index contains an independent samples drawn from the input
    distribution.

    Args:
        logits: 2-D Tensor with shape (batch_size, num_classes). Each row
            should define a categorical distibution with the unnormalized
            log-probabilities for all classes.
        num_samples: Int, the number of independent samples to draw for each
            row of the input. This will be the second dimension of the output
            tensor's shape.
        dtype: Optional dtype of the output tensor.
        seed: A Python integer or instance of
            `keras_core.random.SeedGenerator`.
            Used to make the behavior of the initializer
            deterministic. Note that an initializer seeded with an integer
            or None (unseeded) will produce the same random values
            across multiple calls. To get different random values
            across multiple calls, use as seed an instance
            of `keras_core.random.SeedGenerator`.

    Returns:
        A 2-D tensor with (batch_size, num_samples).
    """
    logits_shape = list(backend.convert_to_tensor(logits).shape)
    if len(logits_shape) != 2:
        raise ValueError(
            "`logits` should be a 2-D tensor with shape "
            f"[batch_size, num_classes]. Received: logits={logits}"
        )
    return backend.random.categorical(
        logits, num_samples, dtype=dtype, seed=seed
    )


@keras_core_export("keras_core.random.uniform")
def uniform(shape, minval=0.0, maxval=1.0, dtype=None, seed=None):
    """Draw samples from a uniform distribution.

    The generated values follow a uniform distribution in the range
    `[minval, maxval)`. The lower bound `minval` is included in the range,
    while the upper bound `maxval` is excluded.

    `dtype` must be a floating point type, the default range is `[0, 1)`.

    Args:
        shape: The shape of the random values to generate.
        minval: Floats, defaults to 0. Lower bound of the range of
            random values to generate (inclusive).
        maxval: Floats, defaults to 1. Upper bound of the range of
            random values to generate (exclusive).
        dtype: Optional dtype of the tensor. Only floating point types are
            supported. If not specified, `keras_core.config.floatx()` is used,
            which defaults to `float32` unless you configured it otherwise (via
            `keras_core.config.set_floatx(float_dtype)`)
        seed: A Python integer or instance of
            `keras_core.random.SeedGenerator`.
            Used to make the behavior of the initializer
            deterministic. Note that an initializer seeded with an integer
            or None (unseeded) will produce the same random values
            across multiple calls. To get different random values
            across multiple calls, use as seed an instance
            of `keras_core.random.SeedGenerator`.
    """
    if dtype and not backend.is_float_dtype(dtype):
        raise ValueError(
            "`keras_core.random.uniform` requires a floating point `dtype`. "
            f"Received: dtype={dtype} "
        )
    return backend.random.uniform(
        shape, minval=minval, maxval=maxval, dtype=dtype, seed=seed
    )


@keras_core_export("keras_core.random.randint")
def randint(shape, minval, maxval, dtype="int32", seed=None):
    """Draw random integers from a uniform distribution.

    The generated values follow a uniform distribution in the range
    `[minval, maxval)`. The lower bound `minval` is included in the range,
    while the upper bound `maxval` is excluded.

    `dtype` must be an integer type.

    Args:
        shape: The shape of the random values to generate.
        minval: Floats, defaults to 0. Lower bound of the range of
            random values to generate (inclusive).
        maxval: Floats, defaults to 1. Upper bound of the range of
            random values to generate (exclusive).
        dtype: Optional dtype of the tensor. Only integer types are
            supported. If not specified, `keras_core.config.floatx()` is used,
            which defaults to `float32` unless you configured it otherwise (via
            `keras_core.config.set_floatx(float_dtype)`)
        seed: A Python integer or instance of
            `keras_core.random.SeedGenerator`.
            Used to make the behavior of the initializer
            deterministic. Note that an initializer seeded with an integer
            or None (unseeded) will produce the same random values
            across multiple calls. To get different random values
            across multiple calls, use as seed an instance
            of `keras_core.random.SeedGenerator`.
    """
    if dtype and not backend.is_int_dtype(dtype):
        raise ValueError(
            "`keras_core.random.randint` requires an integer `dtype`. "
            f"Received: dtype={dtype} "
        )
    return backend.random.randint(
        shape, minval=minval, maxval=maxval, dtype=dtype, seed=seed
    )


@keras_core_export("keras_core.random.truncated_normal")
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
            supported. If not specified, `keras.config.floatx()` is used,
            which defaults to `float32` unless you configured it otherwise (via
            `keras.config.set_floatx(float_dtype)`)
        seed: A Python integer or instance of
            `keras_core.random.SeedGenerator`.
            Used to make the behavior of the initializer
            deterministic. Note that an initializer seeded with an integer
            or None (unseeded) will produce the same random values
            across multiple calls. To get different random values
            across multiple calls, use as seed an instance
            of `keras_core.random.SeedGenerator`.
    """
    return backend.random.truncated_normal(
        shape, mean=mean, stddev=stddev, dtype=dtype, seed=seed
    )


@keras_core_export("keras_core.random.dropout")
def dropout(inputs, rate, noise_shape=None, seed=None):
    return backend.random.dropout(
        inputs, rate, noise_shape=noise_shape, seed=seed
    )
