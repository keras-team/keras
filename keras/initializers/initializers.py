# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Keras initializers."""

import math
import warnings

import tensorflow.compat.v2 as tf

from keras import backend
from keras.dtensor import utils

# isort: off
from tensorflow.python.util.tf_export import keras_export

_PARTITION_SHAPE = "partition_shape"
_PARTITION_OFFSET = "partition_offset"
_LAYOUT = "layout"
_ALLOWED_INITIALIZER_KWARGS = [_PARTITION_SHAPE, _PARTITION_OFFSET, _LAYOUT]


@keras_export("keras.initializers.Initializer")
class Initializer:
    """Initializer base class: all Keras initializers inherit from this class.

    Initializers should implement a `__call__` method with the following
    signature:

    ```python
    def __call__(self, shape, dtype=None, **kwargs):
      # returns a tensor of shape `shape` and dtype `dtype`
      # containing values drawn from a distribution of your choice.
    ```

    Optionally, you an also implement the method `get_config` and the class
    method `from_config` in order to support serialization -- just like with
    any Keras object.

    Here's a simple example: a random normal initializer.

    ```python
    import tensorflow as tf

    class ExampleRandomNormal(tf.keras.initializers.Initializer):

      def __init__(self, mean, stddev):
        self.mean = mean
        self.stddev = stddev

      def __call__(self, shape, dtype=None, **kwargs):
        return tf.random.normal(
            shape, mean=self.mean, stddev=self.stddev, dtype=dtype)

      def get_config(self):  # To support serialization
        return {"mean": self.mean, "stddev": self.stddev}
    ```

    Note that we don't have to implement `from_config` in the example above
    since the constructor arguments of the class the keys in the config returned
    by `get_config` are the same. In this case, the default `from_config` works
    fine.
    """

    def __call__(self, shape, dtype=None, **kwargs):
        """Returns a tensor object initialized as specified by the initializer.

        Args:
          shape: Shape of the tensor.
          dtype: Optional dtype of the tensor.
          **kwargs: Additional keyword arguments.
        """
        raise NotImplementedError(
            "Initializer subclasses must implement the `__call__()` method."
        )

    def get_config(self):
        """Returns the initializer's configuration as a JSON-serializable dict.

        Returns:
          A JSON-serializable Python dict.
        """
        return {}

    @classmethod
    def from_config(cls, config):
        """Instantiates an initializer from a configuration dictionary.

        Example:

        ```python
        initializer = RandomUniform(-1, 1)
        config = initializer.get_config()
        initializer = RandomUniform.from_config(config)
        ```

        Args:
          config: A Python dictionary, the output of `get_config`.

        Returns:
          A `tf.keras.initializers.Initializer` instance.
        """
        config.pop("dtype", None)
        return cls(**config)

    def _warn_reuse(self):
        if getattr(self, "_used", False):
            if getattr(self, "seed", None) is None:
                warnings.warn(
                    f"The initializer {self.__class__.__name__} is unseeded "
                    "and being called multiple times, which will return "
                    "identical values each time (even if the initializer is "
                    "unseeded). Please update your code to provide a seed to "
                    "the initializer, or avoid using the same initalizer "
                    "instance more than once."
                )
        else:
            self._used = True


@keras_export("keras.initializers.Zeros", "keras.initializers.zeros", v1=[])
class Zeros(Initializer):
    """Initializer that generates tensors initialized to 0.

    Also available via the shortcut function `tf.keras.initializers.zeros`.

    Examples:

    >>> # Standalone usage:
    >>> initializer = tf.keras.initializers.Zeros()
    >>> values = initializer(shape=(2, 2))

    >>> # Usage in a Keras layer:
    >>> initializer = tf.keras.initializers.Zeros()
    >>> layer = tf.keras.layers.Dense(3, kernel_initializer=initializer)
    """

    def __call__(self, shape, dtype=None, **kwargs):
        """Returns a tensor object initialized as specified by the initializer.

        Args:
          shape: Shape of the tensor.
          dtype: Optional dtype of the tensor. Only numeric or boolean dtypes
            are supported. If not specified, `tf.keras.backend.floatx()` is
            used, which default to `float32` unless you configured it otherwise
            (via `tf.keras.backend.set_floatx(float_dtype)`).
          **kwargs: Additional keyword arguments.
        """
        _validate_kwargs(self.__class__.__name__, kwargs)
        dtype = _get_dtype(dtype)
        if not dtype.is_numpy_compatible or dtype == tf.string:
            raise ValueError(f"Expected numeric or boolean dtype, got {dtype}.")
        if _PARTITION_SHAPE in kwargs:
            shape = kwargs[_PARTITION_SHAPE]
        layout = kwargs.pop("layout", None)
        if layout:
            return utils.call_with_layout(
                tf.zeros, layout, shape=shape, dtype=dtype
            )
        return tf.zeros(shape, dtype)


@keras_export("keras.initializers.Ones", "keras.initializers.ones", v1=[])
class Ones(Initializer):
    """Initializer that generates tensors initialized to 1.

    Also available via the shortcut function `tf.keras.initializers.ones`.

    Examples:

    >>> # Standalone usage:
    >>> initializer = tf.keras.initializers.Ones()
    >>> values = initializer(shape=(2, 2))

    >>> # Usage in a Keras layer:
    >>> initializer = tf.keras.initializers.Ones()
    >>> layer = tf.keras.layers.Dense(3, kernel_initializer=initializer)
    """

    def __call__(self, shape, dtype=None, **kwargs):
        """Returns a tensor object initialized as specified by the initializer.

        Args:
          shape: Shape of the tensor.
          dtype: Optional dtype of the tensor. Only numeric or boolean dtypes
            are supported. If not specified, `tf.keras.backend.floatx()` is
            used, which default to `float32` unless you configured it otherwise
            (via `tf.keras.backend.set_floatx(float_dtype)`).
          **kwargs: Additional keyword arguments.
        """
        _validate_kwargs(self.__class__.__name__, kwargs)
        dtype = _get_dtype(dtype)
        if not dtype.is_numpy_compatible or dtype == tf.string:
            raise ValueError(f"Expected numeric or boolean dtype, got {dtype}.")
        if _PARTITION_SHAPE in kwargs:
            shape = kwargs[_PARTITION_SHAPE]
        layout = kwargs.pop("layout", None)
        if layout:
            return utils.call_with_layout(
                tf.ones, layout, shape=shape, dtype=dtype
            )
        return tf.ones(shape, dtype)


@keras_export(
    "keras.initializers.Constant", "keras.initializers.constant", v1=[]
)
class Constant(Initializer):
    """Initializer that generates tensors with constant values.

    Also available via the shortcut function `tf.keras.initializers.constant`.

    Only scalar values are allowed.
    The constant value provided must be convertible to the dtype requested
    when calling the initializer.

    Examples:

    >>> # Standalone usage:
    >>> initializer = tf.keras.initializers.Constant(3.)
    >>> values = initializer(shape=(2, 2))

    >>> # Usage in a Keras layer:
    >>> initializer = tf.keras.initializers.Constant(3.)
    >>> layer = tf.keras.layers.Dense(3, kernel_initializer=initializer)

    Args:
      value: A Python scalar.
    """

    def __init__(self, value=0):
        self.value = value

    def __call__(self, shape, dtype=None, **kwargs):
        """Returns a tensor object initialized to `self.value`.

        Args:
          shape: Shape of the tensor.
          dtype: Optional dtype of the tensor. If not specified,
           `tf.keras.backend.floatx()` is used,
           which default to `float32` unless you configured it otherwise
           (via `tf.keras.backend.set_floatx(float_dtype)`).
          **kwargs: Additional keyword arguments.
        """
        _validate_kwargs(self.__class__.__name__, kwargs)
        dtype = _get_dtype(dtype)
        if _PARTITION_SHAPE in kwargs:
            shape = kwargs[_PARTITION_SHAPE]
        layout = kwargs.pop("layout", None)
        if layout:
            return utils.call_with_layout(
                tf.constant, layout, self.value, shape=shape, dtype=dtype
            )
        return tf.constant(self.value, dtype=_get_dtype(dtype), shape=shape)

    def get_config(self):
        return {"value": self.value}


@keras_export(
    "keras.initializers.RandomUniform",
    "keras.initializers.random_uniform",
    v1=[],
)
class RandomUniform(Initializer):
    """Initializer that generates tensors with a uniform distribution.

    Also available via the shortcut function
    `tf.keras.initializers.random_uniform`.

    Examples:

    >>> # Standalone usage:
    >>> initializer = tf.keras.initializers.RandomUniform(minval=0., maxval=1.)
    >>> values = initializer(shape=(2, 2))

    >>> # Usage in a Keras layer:
    >>> initializer = tf.keras.initializers.RandomUniform(minval=0., maxval=1.)
    >>> layer = tf.keras.layers.Dense(3, kernel_initializer=initializer)

    Args:
      minval: A python scalar or a scalar tensor. Lower bound of the range of
        random values to generate (inclusive).
      maxval: A python scalar or a scalar tensor. Upper bound of the range of
        random values to generate (exclusive).
      seed: A Python integer. Used to make the behavior of the initializer
        deterministic. Note that a seeded initializer will produce the same
        random values across multiple calls.
    """

    def __init__(self, minval=-0.05, maxval=0.05, seed=None):
        self.minval = minval
        self.maxval = maxval
        self.seed = seed
        self._random_generator = backend.RandomGenerator(
            seed, rng_type="stateless"
        )

    def __call__(self, shape, dtype=None, **kwargs):
        """Returns a tensor object initialized as specified by the initializer.

        Args:
          shape: Shape of the tensor.
          dtype: Optional dtype of the tensor. Only floating point and integer
          types are supported. If not specified,
            `tf.keras.backend.floatx()` is used,
           which default to `float32` unless you configured it otherwise
           (via `tf.keras.backend.set_floatx(float_dtype)`).
          **kwargs: Additional keyword arguments.
        """
        _validate_kwargs(self.__class__.__name__, kwargs)
        dtype = _get_dtype(dtype)
        if not dtype.is_floating and not dtype.is_integer:
            raise ValueError(f"Expected float or integer dtype, got {dtype}.")
        if _PARTITION_SHAPE in kwargs:
            shape = kwargs[_PARTITION_SHAPE]
        partition_offset = kwargs.get(_PARTITION_OFFSET, None)
        if partition_offset is None:
            # We skip the reuse warning for partitioned variable, since the same
            # initializer will be called multiple times for each partition.
            self._warn_reuse()
        nonce = hash(partition_offset) if partition_offset else None
        layout = kwargs.pop("layout", None)
        if layout:
            _ensure_keras_seeded()
            return utils.call_with_layout(
                self._random_generator.random_uniform,
                layout,
                shape,
                self.minval,
                self.maxval,
                dtype,
                nonce,
            )
        return self._random_generator.random_uniform(
            shape, self.minval, self.maxval, dtype, nonce
        )

    def get_config(self):
        return {"minval": self.minval, "maxval": self.maxval, "seed": self.seed}


@keras_export(
    "keras.initializers.RandomNormal", "keras.initializers.random_normal", v1=[]
)
class RandomNormal(Initializer):
    """Initializer that generates tensors with a normal distribution.

    Also available via the shortcut function
    `tf.keras.initializers.random_normal`.

    Examples:

    >>> # Standalone usage:
    >>> initializer = tf.keras.initializers.RandomNormal(mean=0., stddev=1.)
    >>> values = initializer(shape=(2, 2))

    >>> # Usage in a Keras layer:
    >>> initializer = tf.keras.initializers.RandomNormal(mean=0., stddev=1.)
    >>> layer = tf.keras.layers.Dense(3, kernel_initializer=initializer)

    Args:
      mean: a python scalar or a scalar tensor. Mean of the random values to
        generate.
      stddev: a python scalar or a scalar tensor. Standard deviation of the
        random values to generate.
      seed: A Python integer. Used to make the behavior of the initializer
        deterministic. Note that a seeded initializer will produce the same
        random values across multiple calls.
    """

    def __init__(self, mean=0.0, stddev=0.05, seed=None):
        self.mean = mean
        self.stddev = stddev
        self.seed = seed
        self._random_generator = backend.RandomGenerator(
            seed, rng_type="stateless"
        )

    def __call__(self, shape, dtype=None, **kwargs):
        """Returns a tensor object initialized to random normal values.

        Args:
          shape: Shape of the tensor.
          dtype: Optional dtype of the tensor. Only floating point types are
            supported. If not specified, `tf.keras.backend.floatx()` is used,
            which default to `float32` unless you configured it otherwise (via
            `tf.keras.backend.set_floatx(float_dtype)`)
          **kwargs: Additional keyword arguments.
        """
        _validate_kwargs(self.__class__.__name__, kwargs)
        dtype = _assert_float_dtype(_get_dtype(dtype))
        if _PARTITION_SHAPE in kwargs:
            shape = kwargs[_PARTITION_SHAPE]
        partition_offset = kwargs.get(_PARTITION_OFFSET, None)
        if partition_offset is None:
            # We skip the reuse warning for partitioned variable, since the same
            # initializer will be called multiple times for each partition.
            self._warn_reuse()
        nonce = hash(partition_offset) if partition_offset else None
        layout = kwargs.pop("layout", None)
        if layout:
            _ensure_keras_seeded()
            return utils.call_with_layout(
                self._random_generator.random_normal,
                layout,
                shape,
                self.mean,
                self.stddev,
                dtype,
                nonce,
            )
        return self._random_generator.random_normal(
            shape, self.mean, self.stddev, dtype, nonce
        )

    def get_config(self):
        return {"mean": self.mean, "stddev": self.stddev, "seed": self.seed}


@keras_export(
    "keras.initializers.TruncatedNormal",
    "keras.initializers.truncated_normal",
    v1=[],
)
class TruncatedNormal(Initializer):
    """Initializer that generates a truncated normal distribution.

    Also available via the shortcut function
    `tf.keras.initializers.truncated_normal`.

    The values generated are similar to values from a
    `tf.keras.initializers.RandomNormal` initializer except that values more
    than two standard deviations from the mean are
    discarded and re-drawn.

    Examples:

    >>> # Standalone usage:
    >>> initializer = tf.keras.initializers.TruncatedNormal(mean=0., stddev=1.)
    >>> values = initializer(shape=(2, 2))

    >>> # Usage in a Keras layer:
    >>> initializer = tf.keras.initializers.TruncatedNormal(mean=0., stddev=1.)
    >>> layer = tf.keras.layers.Dense(3, kernel_initializer=initializer)

    Args:
      mean: a python scalar or a scalar tensor. Mean of the random values
        to generate.
      stddev: a python scalar or a scalar tensor. Standard deviation of the
        random values to generate before truncation.
      seed: A Python integer. Used to make the behavior of the initializer
        deterministic. Note that a seeded initializer will produce the same
        random values across multiple calls.
    """

    def __init__(self, mean=0.0, stddev=0.05, seed=None):
        self.mean = mean
        self.stddev = stddev
        self.seed = seed
        self._random_generator = backend.RandomGenerator(
            seed, rng_type="stateless"
        )

    def __call__(self, shape, dtype=None, **kwargs):
        """Returns a tensor initialized to random normal values (truncated).

        Args:
          shape: Shape of the tensor.
          dtype: Optional dtype of the tensor. Only floating point types are
            supported. If not specified, `tf.keras.backend.floatx()` is used,
            which default to `float32` unless you configured it otherwise (via
            `tf.keras.backend.set_floatx(float_dtype)`)
          **kwargs: Additional keyword arguments.
        """
        _validate_kwargs(self.__class__.__name__, kwargs)
        dtype = _assert_float_dtype(_get_dtype(dtype))
        if _PARTITION_SHAPE in kwargs:
            shape = kwargs[_PARTITION_SHAPE]
        partition_offset = kwargs.get(_PARTITION_OFFSET, None)
        if partition_offset is None:
            # We skip the reuse warning for partitioned variable, since the same
            # initializer will be called multiple times for each partition.
            self._warn_reuse()
        nonce = hash(partition_offset) if partition_offset else None
        layout = kwargs.pop("layout", None)
        if layout:
            # TODO(scottzhu): Remove this once the forward compat period above
            # is expired.
            self._random_generator._rng_type = (
                self._random_generator.RNG_STATEFUL
            )
            _ensure_keras_seeded()
            return utils.call_with_layout(
                self._random_generator.truncated_normal,
                layout,
                shape,
                self.mean,
                self.stddev,
                dtype,
                nonce,
            )
        return self._random_generator.truncated_normal(
            shape, self.mean, self.stddev, dtype, nonce
        )

    def get_config(self):
        return {"mean": self.mean, "stddev": self.stddev, "seed": self.seed}


@keras_export(
    "keras.initializers.VarianceScaling",
    "keras.initializers.variance_scaling",
    v1=[],
)
class VarianceScaling(Initializer):
    """Initializer that adapts its scale to the shape of its input tensors.

    Also available via the shortcut function
    `tf.keras.initializers.variance_scaling`.

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
    >>> initializer = tf.keras.initializers.VarianceScaling(
    ... scale=0.1, mode='fan_in', distribution='uniform')
    >>> values = initializer(shape=(2, 2))

    >>> # Usage in a Keras layer:
    >>> initializer = tf.keras.initializers.VarianceScaling(
    ... scale=0.1, mode='fan_in', distribution='uniform')
    >>> layer = tf.keras.layers.Dense(3, kernel_initializer=initializer)

    Args:
      scale: Scaling factor (positive float).
      mode: One of "fan_in", "fan_out", "fan_avg".
      distribution: Random distribution to use. One of "truncated_normal",
        "untruncated_normal" and  "uniform".
      seed: A Python integer. Used to make the behavior of the initializer
        deterministic. Note that a seeded initializer will produce the same
        random values across multiple calls.
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
                f"`scale` must be positive float. Received: scale={scale}."
            )
        allowed_modes = {"fan_in", "fan_out", "fan_avg"}
        if mode not in allowed_modes:
            raise ValueError(
                f"Invalid `mode` argument: {mode}. "
                f"Please use one of the {allowed_modes}."
            )
        distribution = distribution.lower()
        # Compatibility with keras-team/keras.
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
                f"Allowed distributions: {allowed_distributions}."
            )
        self.scale = scale
        self.mode = mode
        self.distribution = distribution
        self.seed = seed
        self._random_generator = backend.RandomGenerator(
            seed, rng_type="stateless"
        )

    def __call__(self, shape, dtype=None, **kwargs):
        """Returns a tensor object initialized as specified by the initializer.

        Args:
          shape: Shape of the tensor.
          dtype: Optional dtype of the tensor. Only floating point types are
            supported. If not specified, `tf.keras.backend.floatx()` is used,
            which default to `float32` unless you configured it otherwise (via
            `tf.keras.backend.set_floatx(float_dtype)`)
          **kwargs: Additional keyword arguments.
        """
        _validate_kwargs(self.__class__.__name__, kwargs)
        dtype = _assert_float_dtype(_get_dtype(dtype))
        if _PARTITION_SHAPE in kwargs:
            shape = kwargs[_PARTITION_SHAPE]
        partition_offset = kwargs.get(_PARTITION_OFFSET, None)
        if partition_offset is None:
            # We skip the reuse warning for partitioned variable, since the same
            # initializer will be called multiple times for each partition.
            self._warn_reuse()
        nonce = hash(partition_offset) if partition_offset else None
        layout = kwargs.pop("layout", None)
        if layout:
            _ensure_keras_seeded()
            return utils.call_with_layout(
                self._generate_init_val,
                layout,
                shape=shape,
                dtype=dtype,
                nonce=nonce,
            )
        return self._generate_init_val(shape=shape, dtype=dtype, nonce=nonce)

    def _generate_init_val(self, shape, dtype, nonce):
        scale = self.scale
        fan_in, fan_out = _compute_fans(shape)
        if self.mode == "fan_in":
            scale /= max(1.0, fan_in)
        elif self.mode == "fan_out":
            scale /= max(1.0, fan_out)
        else:
            scale /= max(1.0, (fan_in + fan_out) / 2.0)
        if self.distribution == "truncated_normal":
            # constant from scipy.stats.truncnorm.std(a=-2, b=2, loc=0.,
            # scale=1.)
            stddev = math.sqrt(scale) / 0.87962566103423978
            return self._random_generator.truncated_normal(
                shape, 0.0, stddev, dtype, nonce
            )
        elif self.distribution == "untruncated_normal":
            stddev = math.sqrt(scale)
            return self._random_generator.random_normal(
                shape, 0.0, stddev, dtype, nonce
            )
        else:
            limit = math.sqrt(3.0 * scale)
            return self._random_generator.random_uniform(
                shape, -limit, limit, dtype, nonce
            )

    def get_config(self):
        return {
            "scale": self.scale,
            "mode": self.mode,
            "distribution": self.distribution,
            "seed": self.seed,
        }


@keras_export(
    "keras.initializers.Orthogonal", "keras.initializers.orthogonal", v1=[]
)
class Orthogonal(Initializer):
    """Initializer that generates an orthogonal matrix.

    Also available via the shortcut function `tf.keras.initializers.orthogonal`.

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
    >>> initializer = tf.keras.initializers.Orthogonal()
    >>> values = initializer(shape=(2, 2))

    >>> # Usage in a Keras layer:
    >>> initializer = tf.keras.initializers.Orthogonal()
    >>> layer = tf.keras.layers.Dense(3, kernel_initializer=initializer)

    Args:
      gain: multiplicative factor to apply to the orthogonal matrix
      seed: A Python integer. Used to make the behavior of the initializer
        deterministic. Note that a seeded initializer will produce the same
        random values across multiple calls.

    References:
      - [Saxe et al., 2014](https://openreview.net/forum?id=_wzZwKpTDF_9C)
    """

    def __init__(self, gain=1.0, seed=None):
        self.gain = gain
        self.seed = seed
        self._random_generator = backend.RandomGenerator(
            seed, rng_type="stateless"
        )

    def __call__(self, shape, dtype=None, **kwargs):
        """Returns a tensor object initialized to an orthogonal matrix.

        Args:
          shape: Shape of the tensor.
          dtype: Optional dtype of the tensor. Only floating point types are
            supported. If not specified, `tf.keras.backend.floatx()` is used,
           which default to `float32` unless you configured it otherwise
           (via `tf.keras.backend.set_floatx(float_dtype)`)
          **kwargs: Additional keyword arguments.
        """
        _validate_kwargs(
            self.__class__.__name__, kwargs, support_partition=False
        )
        dtype = _assert_float_dtype(_get_dtype(dtype))
        # Check the shape
        if len(shape) < 2:
            raise ValueError(
                "The tensor to initialize must be "
                "at least two-dimensional. Received: "
                f"shape={shape} of rank {len(shape)}."
            )
        self._warn_reuse()
        layout = kwargs.pop("layout", None)
        if layout:
            _ensure_keras_seeded()
            return utils.call_with_layout(
                self._generate_init_val, layout, shape=shape, dtype=dtype
            )
        return self._generate_init_val(shape, dtype)

    def _generate_init_val(self, shape, dtype):
        # Flatten the input shape with the last dimension remaining
        # its original shape so it works for conv2d
        num_rows = 1
        for dim in shape[:-1]:
            num_rows *= dim
        num_cols = shape[-1]
        flat_shape = (max(num_cols, num_rows), min(num_cols, num_rows))

        # Generate a random matrix
        a = self._random_generator.random_normal(flat_shape, dtype=dtype)
        # Compute the qr factorization
        q, r = tf.linalg.qr(a, full_matrices=False)
        # Make Q uniform
        d = tf.linalg.tensor_diag_part(r)
        q *= tf.sign(d)
        if num_rows < num_cols:
            q = tf.linalg.matrix_transpose(q)
        return self.gain * tf.reshape(q, shape)

    def get_config(self):
        return {"gain": self.gain, "seed": self.seed}


@keras_export(
    "keras.initializers.Identity", "keras.initializers.identity", v1=[]
)
class Identity(Initializer):
    """Initializer that generates the identity matrix.

    Also available via the shortcut function `tf.keras.initializers.identity`.

    Only usable for generating 2D matrices.

    Examples:

    >>> # Standalone usage:
    >>> initializer = tf.keras.initializers.Identity()
    >>> values = initializer(shape=(2, 2))

    >>> # Usage in a Keras layer:
    >>> initializer = tf.keras.initializers.Identity()
    >>> layer = tf.keras.layers.Dense(3, kernel_initializer=initializer)

    Args:
      gain: Multiplicative factor to apply to the identity matrix.
    """

    def __init__(self, gain=1.0):
        self.gain = gain

    def __call__(self, shape, dtype=None, **kwargs):
        """Returns a tensor object initialized to a 2D identity matrix.

        Args:
          shape: Shape of the tensor. It should have exactly rank 2.
          dtype: Optional dtype of the tensor. Only floating point types are
           supported. If not specified, `tf.keras.backend.floatx()` is used,
           which default to `float32` unless you configured it otherwise
           (via `tf.keras.backend.set_floatx(float_dtype)`)
          **kwargs: Additional keyword arguments.
        """
        _validate_kwargs(
            self.__class__.__name__, kwargs, support_partition=False
        )
        dtype = _assert_float_dtype(_get_dtype(dtype))
        if len(shape) != 2:
            raise ValueError(
                "Identity matrix initializer can only be used for 2D matrices. "
                f"Received: shape={shape} of rank {len(shape)}."
            )
        layout = kwargs.pop("layout", None)
        if layout:
            return utils.call_with_layout(
                self._generate_init_val, layout, shape=shape, dtype=dtype
            )
        return self._generate_init_val(shape, dtype)

    def _generate_init_val(self, shape, dtype):
        initializer = tf.eye(*shape, dtype=dtype)
        return self.gain * initializer

    def get_config(self):
        return {"gain": self.gain}


@keras_export(
    "keras.initializers.GlorotUniform",
    "keras.initializers.glorot_uniform",
    v1=[],
)
class GlorotUniform(VarianceScaling):
    """The Glorot uniform initializer, also called Xavier uniform initializer.

    Also available via the shortcut function
    `tf.keras.initializers.glorot_uniform`.

    Draws samples from a uniform distribution within `[-limit, limit]`, where
    `limit = sqrt(6 / (fan_in + fan_out))` (`fan_in` is the number of input
    units in the weight tensor and `fan_out` is the number of output units).

    Examples:

    >>> # Standalone usage:
    >>> initializer = tf.keras.initializers.GlorotUniform()
    >>> values = initializer(shape=(2, 2))

    >>> # Usage in a Keras layer:
    >>> initializer = tf.keras.initializers.GlorotUniform()
    >>> layer = tf.keras.layers.Dense(3, kernel_initializer=initializer)

    Args:
      seed: A Python integer. Used to make the behavior of the initializer
        deterministic. Note that a seeded initializer will not produce the same
        random values across multiple calls, but multiple initializers will
        produce the same sequence when constructed with the same seed value.

    References:
      - [Glorot et al., 2010](http://proceedings.mlr.press/v9/glorot10a.html)
    """

    def __init__(self, seed=None):
        super().__init__(
            scale=1.0, mode="fan_avg", distribution="uniform", seed=seed
        )

    def get_config(self):
        return {"seed": self.seed}


@keras_export(
    "keras.initializers.GlorotNormal", "keras.initializers.glorot_normal", v1=[]
)
class GlorotNormal(VarianceScaling):
    """The Glorot normal initializer, also called Xavier normal initializer.

    Also available via the shortcut function
    `tf.keras.initializers.glorot_normal`.

    Draws samples from a truncated normal distribution centered on 0 with
    `stddev = sqrt(2 / (fan_in + fan_out))` where `fan_in` is the number of
    input units in the weight tensor and `fan_out` is the number of output units
    in the weight tensor.

    Examples:

    >>> # Standalone usage:
    >>> initializer = tf.keras.initializers.GlorotNormal()
    >>> values = initializer(shape=(2, 2))

    >>> # Usage in a Keras layer:
    >>> initializer = tf.keras.initializers.GlorotNormal()
    >>> layer = tf.keras.layers.Dense(3, kernel_initializer=initializer)

    Args:
      seed: A Python integer. Used to make the behavior of the initializer
        deterministic. Note that a seeded initializer will not produce the same
        random values across multiple calls, but multiple initializers will
        produce the same sequence when constructed with the same seed value.

    References:
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
        return {"seed": self.seed}


@keras_export(
    "keras.initializers.LecunNormal", "keras.initializers.lecun_normal", v1=[]
)
class LecunNormal(VarianceScaling):
    """Lecun normal initializer.

     Also available via the shortcut function
    `tf.keras.initializers.lecun_normal`.

    Initializers allow you to pre-specify an initialization strategy, encoded in
    the Initializer object, without knowing the shape and dtype of the variable
    being initialized.

    Draws samples from a truncated normal distribution centered on 0 with
    `stddev = sqrt(1 / fan_in)` where `fan_in` is the number of input units in
    the weight tensor.

    Examples:

    >>> # Standalone usage:
    >>> initializer = tf.keras.initializers.LecunNormal()
    >>> values = initializer(shape=(2, 2))

    >>> # Usage in a Keras layer:
    >>> initializer = tf.keras.initializers.LecunNormal()
    >>> layer = tf.keras.layers.Dense(3, kernel_initializer=initializer)

    Args:
      seed: A Python integer. Used to make the behavior of the initializer
        deterministic. Note that a seeded initializer will not produce the same
        random values across multiple calls, but multiple initializers will
        produce the same sequence when constructed with the same seed value.

    References:
      - [Klambauer et al., 2017](https://arxiv.org/abs/1706.02515)
    """

    def __init__(self, seed=None):
        super().__init__(
            scale=1.0, mode="fan_in", distribution="truncated_normal", seed=seed
        )

    def get_config(self):
        return {"seed": self.seed}


@keras_export(
    "keras.initializers.LecunUniform", "keras.initializers.lecun_uniform", v1=[]
)
class LecunUniform(VarianceScaling):
    """Lecun uniform initializer.

     Also available via the shortcut function
    `tf.keras.initializers.lecun_uniform`.

    Draws samples from a uniform distribution within `[-limit, limit]`, where
    `limit = sqrt(3 / fan_in)` (`fan_in` is the number of input units in the
    weight tensor).

    Examples:

    >>> # Standalone usage:
    >>> initializer = tf.keras.initializers.LecunUniform()
    >>> values = initializer(shape=(2, 2))

    >>> # Usage in a Keras layer:
    >>> initializer = tf.keras.initializers.LecunUniform()
    >>> layer = tf.keras.layers.Dense(3, kernel_initializer=initializer)

    Args:
      seed: A Python integer. Used to make the behavior of the initializer
        deterministic. Note that a seeded initializer will not produce the same
        random values across multiple calls, but multiple initializers will
        produce the same sequence when constructed with the same seed value.

    References:
      - [Klambauer et al., 2017](https://arxiv.org/abs/1706.02515)
    """

    def __init__(self, seed=None):
        super().__init__(
            scale=1.0, mode="fan_in", distribution="uniform", seed=seed
        )

    def get_config(self):
        return {"seed": self.seed}


@keras_export(
    "keras.initializers.HeNormal", "keras.initializers.he_normal", v1=[]
)
class HeNormal(VarianceScaling):
    """He normal initializer.

     Also available via the shortcut function
    `tf.keras.initializers.he_normal`.

    It draws samples from a truncated normal distribution centered on 0 with
    `stddev = sqrt(2 / fan_in)` where `fan_in` is the number of input units in
    the weight tensor.

    Examples:

    >>> # Standalone usage:
    >>> initializer = tf.keras.initializers.HeNormal()
    >>> values = initializer(shape=(2, 2))

    >>> # Usage in a Keras layer:
    >>> initializer = tf.keras.initializers.HeNormal()
    >>> layer = tf.keras.layers.Dense(3, kernel_initializer=initializer)

    Args:
      seed: A Python integer. Used to make the behavior of the initializer
        deterministic. Note that a seeded initializer will not produce the same
        random values across multiple calls, but multiple initializers will
        produce the same sequence when constructed with the same seed value.

    References:
      - [He et al., 2015](https://arxiv.org/abs/1502.01852)
    """

    def __init__(self, seed=None):
        super().__init__(
            scale=2.0, mode="fan_in", distribution="truncated_normal", seed=seed
        )

    def get_config(self):
        return {"seed": self.seed}


@keras_export(
    "keras.initializers.HeUniform", "keras.initializers.he_uniform", v1=[]
)
class HeUniform(VarianceScaling):
    """He uniform variance scaling initializer.

     Also available via the shortcut function
    `tf.keras.initializers.he_uniform`.

    Draws samples from a uniform distribution within `[-limit, limit]`, where
    `limit = sqrt(6 / fan_in)` (`fan_in` is the number of input units in the
    weight tensor).

    Examples:

    >>> # Standalone usage:
    >>> initializer = tf.keras.initializers.HeUniform()
    >>> values = initializer(shape=(2, 2))

    >>> # Usage in a Keras layer:
    >>> initializer = tf.keras.initializers.HeUniform()
    >>> layer = tf.keras.layers.Dense(3, kernel_initializer=initializer)

    Args:
      seed: A Python integer. Used to make the behavior of the initializer
        deterministic. Note that a seeded initializer will not produce the same
        random values across multiple calls, but multiple initializers will
        produce the same sequence when constructed with the same seed value.

    References:
      - [He et al., 2015](https://arxiv.org/abs/1502.01852)
    """

    def __init__(self, seed=None):
        super().__init__(
            scale=2.0, mode="fan_in", distribution="uniform", seed=seed
        )

    def get_config(self):
        return {"seed": self.seed}


def _get_dtype(dtype):
    if dtype is None:
        dtype = backend.floatx()
    return tf.as_dtype(dtype)


def _assert_float_dtype(dtype):
    """Validate and return floating point type based on `dtype`.

    `dtype` must be a floating point type.

    Args:
      dtype: The data type to validate.

    Returns:
      Validated type.

    Raises:
      ValueError: if `dtype` is not a floating point type.
    """
    dtype = tf.as_dtype(dtype)
    if not dtype.is_floating:
        raise ValueError(f"Expected floating point type, got {dtype}.")
    return dtype


def _compute_fans(shape):
    """Computes the number of input and output units for a weight shape.

    Args:
      shape: Integer shape tuple or TF tensor shape.

    Returns:
      A tuple of integer scalars (fan_in, fan_out).
    """
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


def _validate_kwargs(cls_name, kwargs, support_partition=True):
    invalid_kwargs = [k for k in kwargs if k not in _ALLOWED_INITIALIZER_KWARGS]
    if invalid_kwargs:
        raise TypeError(
            f"Unknown keyword arguments: {invalid_kwargs}. Allowed "
            f"keyword arguments: {_ALLOWED_INITIALIZER_KWARGS}."
        )
    if not support_partition and (
        _PARTITION_SHAPE in kwargs or _PARTITION_OFFSET in kwargs
    ):
        raise ValueError(
            f"{cls_name} initializer doesn't support "
            "partition-related arguments."
        )


def _ensure_keras_seeded():
    """Make sure the keras.backend global seed generator is set.

    This is important for DTensor use case to ensure that each client are
    initialized with same seed for tf.random.Generator, so that the value
    created are in sync among all the clients.
    """
    if not getattr(backend._SEED_GENERATOR, "generator", None):
        raise ValueError(
            "When using DTensor APIs, you need to set the global seed "
            "before using any Keras initializers. Please make sure "
            "to call `tf.keras.utils.set_random_seed()` in your code."
        )
