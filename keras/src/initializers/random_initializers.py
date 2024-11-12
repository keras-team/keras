import math

from keras.src import ops
from keras.src.api_export import keras_export
from keras.src.backend import random
from keras.src.initializers.initializer import Initializer
from keras.src.saving import serialization_lib


class RandomInitializer(Initializer):
    def __init__(self, seed=None):
        self._init_seed = seed
        if seed is None:
            seed = random.make_default_seed()
        elif isinstance(seed, dict):
            seed = serialization_lib.deserialize_keras_object(seed)
        elif not isinstance(seed, (int, random.SeedGenerator)):
            raise ValueError(
                "`seed` argument should be an instance of "
                "`keras.random.SeedGenerator()` or an integer. "
                f"Received: seed={seed}"
            )
        self.seed = seed

    def get_config(self):
        seed_config = serialization_lib.serialize_keras_object(self._init_seed)
        return {"seed": seed_config}


@keras_export(
    [
        "keras.initializers.RandomNormal",
        "keras.initializers.random_normal",
    ]
)
class RandomNormal(RandomInitializer):
    """Random normal initializer.

    Draws samples from a normal distribution for given parameters.

    Examples:

    >>> # Standalone usage:
    >>> initializer = RandomNormal(mean=0.0, stddev=1.0)
    >>> values = initializer(shape=(2, 2))

    >>> # Usage in a Keras layer:
    >>> initializer = RandomNormal(mean=0.0, stddev=1.0)
    >>> layer = Dense(3, kernel_initializer=initializer)

    Args:
        mean: A python scalar or a scalar keras tensor. Mean of the random
            values to generate.
        stddev: A python scalar or a scalar keras tensor. Standard deviation of
           the random values to generate.
        seed: A Python integer or instance of
            `keras.backend.SeedGenerator`.
            Used to make the behavior of the initializer
            deterministic. Note that an initializer seeded with an integer
            or `None` (unseeded) will produce the same random values
            across multiple calls. To get different random values
            across multiple calls, use as seed an instance
            of `keras.backend.SeedGenerator`.
    """

    def __init__(self, mean=0.0, stddev=0.05, seed=None):
        self.mean = mean
        self.stddev = stddev
        super().__init__(seed=seed)

    def __call__(self, shape, dtype=None):
        return random.normal(
            shape=shape,
            mean=self.mean,
            stddev=self.stddev,
            seed=self.seed,
            dtype=dtype,
        )

    def get_config(self):
        base_config = super().get_config()
        config = {"mean": self.mean, "stddev": self.stddev}
        return {**base_config, **config}


@keras_export(
    [
        "keras.initializers.TruncatedNormal",
        "keras.initializers.truncated_normal",
    ]
)
class TruncatedNormal(RandomInitializer):
    """Initializer that generates a truncated normal distribution.

    The values generated are similar to values from a
    `RandomNormal` initializer, except that values more
    than two standard deviations from the mean are
    discarded and re-drawn.

    Examples:

    >>> # Standalone usage:
    >>> initializer = TruncatedNormal(mean=0., stddev=1.)
    >>> values = initializer(shape=(2, 2))

    >>> # Usage in a Keras layer:
    >>> initializer = TruncatedNormal(mean=0., stddev=1.)
    >>> layer = Dense(3, kernel_initializer=initializer)

    Args:
        mean: A python scalar or a scalar keras tensor. Mean of the random
            values to generate.
        stddev: A python scalar or a scalar keras tensor. Standard deviation of
           the random values to generate.
        seed: A Python integer or instance of
            `keras.backend.SeedGenerator`.
            Used to make the behavior of the initializer
            deterministic. Note that an initializer seeded with an integer
            or `None` (unseeded) will produce the same random values
            across multiple calls. To get different random values
            across multiple calls, use as seed an instance
            of `keras.backend.SeedGenerator`.
    """

    def __init__(self, mean=0.0, stddev=0.05, seed=None):
        self.mean = mean
        self.stddev = stddev
        super().__init__(seed=seed)

    def __call__(self, shape, dtype=None):
        return random.truncated_normal(
            shape=shape,
            mean=self.mean,
            stddev=self.stddev,
            seed=self.seed,
            dtype=dtype,
        )

    def get_config(self):
        base_config = super().get_config()
        config = {"mean": self.mean, "stddev": self.stddev}
        return {**base_config, **config}


@keras_export(
    [
        "keras.initializers.RandomUniform",
        "keras.initializers.random_uniform",
    ]
)
class RandomUniform(RandomInitializer):
    """Random uniform initializer.

    Draws samples from a uniform distribution for given parameters.

    Examples:

    >>> # Standalone usage:
    >>> initializer = RandomUniform(minval=0.0, maxval=1.0)
    >>> values = initializer(shape=(2, 2))

    >>> # Usage in a Keras layer:
    >>> initializer = RandomUniform(minval=0.0, maxval=1.0)
    >>> layer = Dense(3, kernel_initializer=initializer)

    Args:
        minval: A python scalar or a scalar keras tensor. Lower bound of the
            range of random values to generate (inclusive).
        maxval: A python scalar or a scalar keras tensor. Upper bound of the
            range of random values to generate (exclusive).
        seed: A Python integer or instance of
            `keras.backend.SeedGenerator`.
            Used to make the behavior of the initializer
            deterministic. Note that an initializer seeded with an integer
            or `None` (unseeded) will produce the same random values
            across multiple calls. To get different random values
            across multiple calls, use as seed an instance
            of `keras.backend.SeedGenerator`.
    """

    def __init__(self, minval=-0.05, maxval=0.05, seed=None):
        self.minval = minval
        self.maxval = maxval
        super().__init__(seed=seed)

    def __call__(self, shape, dtype=None):
        return random.uniform(
            shape=shape,
            minval=self.minval,
            maxval=self.maxval,
            seed=self.seed,
            dtype=dtype,
        )

    def get_config(self):
        base_config = super().get_config()
        config = {"minval": self.minval, "maxval": self.maxval}
        return {**base_config, **config}


@keras_export(
    [
        "keras.initializers.VarianceScaling",
        "keras.initializers.variance_scaling",
    ]
)
class VarianceScaling(RandomInitializer):
    """Initializer that adapts its scale to the shape of its input tensors.

    With `distribution="truncated_normal" or "untruncated_normal"`, samples are
    drawn from a truncated/untruncated normal distribution with a mean of zero
    and a standard deviation (after truncation, if used) `stddev = sqrt(scale /
    n)`, where `n` is:

    - number of input units in the weight tensor, if `mode="fan_in"`
    - number of output units, if `mode="fan_out"`
    - average of the numbers of input and output units, if `mode="fan_avg"`

    With `distribution="uniform"`, samples are drawn from a uniform distribution
    within `[-limit, limit]`, where `limit = sqrt(3 * scale / n)`.

    Examples:

    >>> # Standalone usage:
    >>> initializer = VarianceScaling(
        scale=0.1, mode='fan_in', distribution='uniform')
    >>> values = initializer(shape=(2, 2))

    >>> # Usage in a Keras layer:
    >>> initializer = VarianceScaling(
        scale=0.1, mode='fan_in', distribution='uniform')
    >>> layer = Dense(3, kernel_initializer=initializer)

    Args:
        scale: Scaling factor (positive float).
        mode: One of `"fan_in"`, `"fan_out"`, `"fan_avg"`.
        distribution: Random distribution to use.
            One of `"truncated_normal"`, `"untruncated_normal"`, or `"uniform"`.
        seed: A Python integer or instance of
            `keras.backend.SeedGenerator`.
            Used to make the behavior of the initializer
            deterministic. Note that an initializer seeded with an integer
            or `None` (unseeded) will produce the same random values
            across multiple calls. To get different random values
            across multiple calls, use as seed an instance
            of `keras.backend.SeedGenerator`.
    """

    def __init__(
        self,
        scale=1.0,
        mode="fan_in",
        distribution="truncated_normal",
        seed=None,
    ):
        if scale <= 0.0:
            raise ValueError(
                "Argument `scale` must be positive float. "
                f"Received: scale={scale}"
            )
        allowed_modes = {"fan_in", "fan_out", "fan_avg"}
        if mode not in allowed_modes:
            raise ValueError(
                f"Invalid `mode` argument: {mode}. "
                f"Please use one of {allowed_modes}"
            )
        distribution = distribution.lower()
        if distribution == "normal":
            distribution = "truncated_normal"
        allowed_distributions = {
            "uniform",
            "truncated_normal",
            "untruncated_normal",
        }
        if distribution not in allowed_distributions:
            raise ValueError(
                f"Invalid `distribution` argument: {distribution}."
                f"Please use one of {allowed_distributions}"
            )
        self.scale = scale
        self.mode = mode
        self.distribution = distribution
        super().__init__(seed=seed)

    def __call__(self, shape, dtype=None):
        scale = self.scale
        fan_in, fan_out = compute_fans(shape)
        if self.mode == "fan_in":
            scale /= max(1.0, fan_in)
        elif self.mode == "fan_out":
            scale /= max(1.0, fan_out)
        else:
            scale /= max(1.0, (fan_in + fan_out) / 2.0)
        if self.distribution == "truncated_normal":
            stddev = math.sqrt(scale) / 0.87962566103423978
            return random.truncated_normal(
                shape, mean=0.0, stddev=stddev, dtype=dtype, seed=self.seed
            )
        elif self.distribution == "untruncated_normal":
            stddev = math.sqrt(scale)
            return random.normal(
                shape, mean=0.0, stddev=stddev, dtype=dtype, seed=self.seed
            )
        else:
            limit = math.sqrt(3.0 * scale)
            return random.uniform(
                shape, minval=-limit, maxval=limit, dtype=dtype, seed=self.seed
            )

    def get_config(self):
        base_config = super().get_config()
        config = {
            "scale": self.scale,
            "mode": self.mode,
            "distribution": self.distribution,
        }
        return {**base_config, **config}


@keras_export(
    [
        "keras.initializers.GlorotUniform",
        "keras.initializers.glorot_uniform",
    ]
)
class GlorotUniform(VarianceScaling):
    """The Glorot uniform initializer, also called Xavier uniform initializer.

    Draws samples from a uniform distribution within `[-limit, limit]`, where
    `limit = sqrt(6 / (fan_in + fan_out))` (`fan_in` is the number of input
    units in the weight tensor and `fan_out` is the number of output units).

    Examples:

    >>> # Standalone usage:
    >>> initializer = GlorotUniform()
    >>> values = initializer(shape=(2, 2))

    >>> # Usage in a Keras layer:
    >>> initializer = GlorotUniform()
    >>> layer = Dense(3, kernel_initializer=initializer)

    Args:
        seed: A Python integer or instance of
            `keras.backend.SeedGenerator`.
            Used to make the behavior of the initializer
            deterministic. Note that an initializer seeded with an integer
            or `None` (unseeded) will produce the same random values
            across multiple calls. To get different random values
            across multiple calls, use as seed an instance
            of `keras.backend.SeedGenerator`.

    Reference:

    - [Glorot et al., 2010](http://proceedings.mlr.press/v9/glorot10a.html)
    """

    def __init__(self, seed=None):
        super().__init__(
            scale=1.0, mode="fan_avg", distribution="uniform", seed=seed
        )

    def get_config(self):
        return {
            "seed": serialization_lib.serialize_keras_object(self._init_seed)
        }


@keras_export(
    [
        "keras.initializers.GlorotNormal",
        "keras.initializers.glorot_normal",
    ]
)
class GlorotNormal(VarianceScaling):
    """The Glorot normal initializer, also called Xavier normal initializer.

    Draws samples from a truncated normal distribution centered on 0 with
    `stddev = sqrt(2 / (fan_in + fan_out))` where `fan_in` is the number of
    input units in the weight tensor and `fan_out` is the number of output units
    in the weight tensor.

    Examples:

    >>> # Standalone usage:
    >>> initializer = GlorotNormal()
    >>> values = initializer(shape=(2, 2))

    >>> # Usage in a Keras layer:
    >>> initializer = GlorotNormal()
    >>> layer = Dense(3, kernel_initializer=initializer)

    Args:
        seed: A Python integer or instance of
            `keras.backend.SeedGenerator`.
            Used to make the behavior of the initializer
            deterministic. Note that an initializer seeded with an integer
            or `None` (unseeded) will produce the same random values
            across multiple calls. To get different random values
            across multiple calls, use as seed an instance
            of `keras.backend.SeedGenerator`.

    Reference:

    - [Glorot et al., 2010](http://proceedings.mlr.press/v9/glorot10a.html)
    """

    def __init__(self, seed=None):
        super().__init__(
            scale=1.0,
            mode="fan_avg",
            distribution="truncated_normal",
            seed=seed,
        )

    def get_config(self):
        return {
            "seed": serialization_lib.serialize_keras_object(self._init_seed)
        }


@keras_export(
    [
        "keras.initializers.LecunNormal",
        "keras.initializers.lecun_normal",
    ]
)
class LecunNormal(VarianceScaling):
    """Lecun normal initializer.

    Initializers allow you to pre-specify an initialization strategy, encoded in
    the Initializer object, without knowing the shape and dtype of the variable
    being initialized.

    Draws samples from a truncated normal distribution centered on 0 with
    `stddev = sqrt(1 / fan_in)` where `fan_in` is the number of input units in
    the weight tensor.

    Examples:

    >>> # Standalone usage:
    >>> initializer = LecunNormal()
    >>> values = initializer(shape=(2, 2))

    >>> # Usage in a Keras layer:
    >>> initializer = LecunNormal()
    >>> layer = Dense(3, kernel_initializer=initializer)

    Args:
        seed: A Python integer or instance of
            `keras.backend.SeedGenerator`.
            Used to make the behavior of the initializer
            deterministic. Note that an initializer seeded with an integer
            or `None` (unseeded) will produce the same random values
            across multiple calls. To get different random values
            across multiple calls, use as seed an instance
            of `keras.backend.SeedGenerator`.

    Reference:

    - [Klambauer et al., 2017](https://arxiv.org/abs/1706.02515)
    """

    def __init__(self, seed=None):
        super().__init__(
            scale=1.0, mode="fan_in", distribution="truncated_normal", seed=seed
        )

    def get_config(self):
        return {
            "seed": serialization_lib.serialize_keras_object(self._init_seed)
        }


@keras_export(
    [
        "keras.initializers.LecunUniform",
        "keras.initializers.lecun_uniform",
    ]
)
class LecunUniform(VarianceScaling):
    """Lecun uniform initializer.

    Draws samples from a uniform distribution within `[-limit, limit]`, where
    `limit = sqrt(3 / fan_in)` (`fan_in` is the number of input units in the
    weight tensor).

    Examples:

    >>> # Standalone usage:
    >>> initializer = LecunUniform()
    >>> values = initializer(shape=(2, 2))

    >>> # Usage in a Keras layer:
    >>> initializer = LecunUniform()
    >>> layer = Dense(3, kernel_initializer=initializer)

    Args:
        seed: A Python integer or instance of
            `keras.backend.SeedGenerator`.
            Used to make the behavior of the initializer
            deterministic. Note that an initializer seeded with an integer
            or `None` (unseeded) will produce the same random values
            across multiple calls. To get different random values
            across multiple calls, use as seed an instance
            of `keras.backend.SeedGenerator`.

    Reference:

    - [Klambauer et al., 2017](https://arxiv.org/abs/1706.02515)
    """

    def __init__(self, seed=None):
        super().__init__(
            scale=1.0, mode="fan_in", distribution="uniform", seed=seed
        )

    def get_config(self):
        return {
            "seed": serialization_lib.serialize_keras_object(self._init_seed)
        }


@keras_export(["keras.initializers.HeNormal", "keras.initializers.he_normal"])
class HeNormal(VarianceScaling):
    """He normal initializer.

    It draws samples from a truncated normal distribution centered on 0 with
    `stddev = sqrt(2 / fan_in)` where `fan_in` is the number of input units in
    the weight tensor.

    Examples:

    >>> # Standalone usage:
    >>> initializer = HeNormal()
    >>> values = initializer(shape=(2, 2))

    >>> # Usage in a Keras layer:
    >>> initializer = HeNormal()
    >>> layer = Dense(3, kernel_initializer=initializer)

    Args:
        seed: A Python integer or instance of
            `keras.backend.SeedGenerator`.
            Used to make the behavior of the initializer
            deterministic. Note that an initializer seeded with an integer
            or `None` (unseeded) will produce the same random values
            across multiple calls. To get different random values
            across multiple calls, use as seed an instance
            of `keras.backend.SeedGenerator`.

    Reference:

    - [He et al., 2015](https://arxiv.org/abs/1502.01852)
    """

    def __init__(self, seed=None):
        super().__init__(
            scale=2.0, mode="fan_in", distribution="truncated_normal", seed=seed
        )

    def get_config(self):
        return {
            "seed": serialization_lib.serialize_keras_object(self._init_seed)
        }


@keras_export(["keras.initializers.HeUniform", "keras.initializers.he_uniform"])
class HeUniform(VarianceScaling):
    """He uniform variance scaling initializer.

    Draws samples from a uniform distribution within `[-limit, limit]`, where
    `limit = sqrt(6 / fan_in)` (`fan_in` is the number of input units in the
    weight tensor).

    Examples:

    >>> # Standalone usage:
    >>> initializer = HeUniform()
    >>> values = initializer(shape=(2, 2))

    >>> # Usage in a Keras layer:
    >>> initializer = HeUniform()
    >>> layer = Dense(3, kernel_initializer=initializer)

    Args:
        seed: A Python integer or instance of
            `keras.backend.SeedGenerator`.
            Used to make the behavior of the initializer
            deterministic. Note that an initializer seeded with an integer
            or `None` (unseeded) will produce the same random values
            across multiple calls. To get different random values
            across multiple calls, use as seed an instance
            of `keras.backend.SeedGenerator`.

    Reference:

    - [He et al., 2015](https://arxiv.org/abs/1502.01852)
    """

    def __init__(self, seed=None):
        super().__init__(
            scale=2.0, mode="fan_in", distribution="uniform", seed=seed
        )

    def get_config(self):
        return {
            "seed": serialization_lib.serialize_keras_object(self._init_seed)
        }


def compute_fans(shape):
    """Computes the number of input and output units for a weight shape.

    Args:
        shape: Integer shape tuple.

    Returns:
        A tuple of integer scalars: `(fan_in, fan_out)`.
    """
    shape = tuple(shape)
    if len(shape) < 1:  # Just to avoid errors for constants.
        fan_in = fan_out = 1
    elif len(shape) == 1:
        fan_in = fan_out = shape[0]
    elif len(shape) == 2:
        fan_in = shape[0]
        fan_out = shape[1]
    else:
        # Assuming convolution kernels (2D, 3D, or more).
        # kernel shape: (..., input_depth, depth)
        receptive_field_size = 1
        for dim in shape[:-2]:
            receptive_field_size *= dim
        fan_in = shape[-2] * receptive_field_size
        fan_out = shape[-1] * receptive_field_size
    return int(fan_in), int(fan_out)


@keras_export(
    [
        "keras.initializers.Orthogonal",
        "keras.initializers.orthogonal",
        "keras.initializers.OrthogonalInitializer",
    ]
)
class Orthogonal(RandomInitializer):
    """Initializer that generates an orthogonal matrix.

    If the shape of the tensor to initialize is two-dimensional, it is
    initialized with an orthogonal matrix obtained from the QR decomposition of
    a matrix of random numbers drawn from a normal distribution. If the matrix
    has fewer rows than columns then the output will have orthogonal rows.
    Otherwise, the output will have orthogonal columns.

    If the shape of the tensor to initialize is more than two-dimensional,
    a matrix of shape `(shape[0] * ... * shape[n - 2], shape[n - 1])`
    is initialized, where `n` is the length of the shape vector.
    The matrix is subsequently reshaped to give a tensor of the desired shape.

    Examples:

    >>> # Standalone usage:
    >>> initializer = keras.initializers.Orthogonal()
    >>> values = initializer(shape=(2, 2))

    >>> # Usage in a Keras layer:
    >>> initializer = keras.initializers.Orthogonal()
    >>> layer = keras.layers.Dense(3, kernel_initializer=initializer)

    Args:
        gain: Multiplicative factor to apply to the orthogonal matrix.
        seed: A Python integer. Used to make the behavior of the initializer
            deterministic.

    Reference:

    - [Saxe et al., 2014](https://openreview.net/forum?id=_wzZwKpTDF_9C)
    """

    def __init__(self, gain=1.0, seed=None):
        self.gain = gain
        super().__init__(seed=seed)

    def __call__(self, shape, dtype=None):
        if len(shape) < 2:
            raise ValueError(
                "The tensor to initialize must be "
                "at least two-dimensional. Received: "
                f"shape={shape} of rank {len(shape)}."
            )

        # Flatten the input shape with the last dimension remaining
        # its original shape so it works for conv2d
        num_rows = 1
        for dim in shape[:-1]:
            num_rows *= dim
        num_cols = shape[-1]
        flat_shape = (max(num_cols, num_rows), min(num_cols, num_rows))

        # Generate a random matrix
        a = random.normal(flat_shape, seed=self.seed, dtype=dtype)
        # Compute the qr factorization
        q, r = ops.qr(a)
        # Make Q uniform
        d = ops.diag(r)
        q *= ops.sign(d)
        if num_rows < num_cols:
            q = ops.transpose(q)
        return self.gain * ops.reshape(q, shape)

    def get_config(self):
        base_config = super().get_config()
        config = {"gain": self.gain}
        return {**base_config, **config}
