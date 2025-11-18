import math

import numpy as np

from keras.src import backend
from keras.src import ops
from keras.src.api_export import keras_export
from keras.src.layers.preprocessing.data_layer import DataLayer
from keras.src.utils.module_utils import tensorflow as tf


@keras_export("keras.layers.Normalization")
class Normalization(DataLayer):
    """A preprocessing layer that normalizes continuous features.

    This layer will shift and scale inputs into a distribution centered around
    0 with standard deviation 1. It accomplishes this by precomputing the mean
    and variance of the data, and calling `(input - mean) / sqrt(var)` at
    runtime.

    The mean and variance values for the layer must be either supplied on
    construction or learned via `adapt()`. `adapt()` will compute the mean and
    variance of the data and store them as the layer's weights. `adapt()` should
    be called before `fit()`, `evaluate()`, or `predict()`.

    **Note:** This layer is safe to use inside a `tf.data` or `grain` pipeline
    (independently of which backend you're using).

    Args:
        axis: Integer, tuple of integers, or None. The axis or axes that should
            have a separate mean and variance for each index in the shape.
            For example, if shape is `(None, 5)` and `axis=1`, the layer will
            track 5 separate mean and variance values for the last axis.
            If `axis` is set to `None`, the layer will normalize
            all elements in the input by a scalar mean and variance.
            When `-1`, the last axis of the input is assumed to be a
            feature dimension and is normalized per index.
            Note that in the specific case of batched scalar inputs where
            the only axis is the batch axis, the default will normalize
            each index in the batch separately.
            In this case, consider passing `axis=None`. Defaults to `-1`.
        mean: The mean value(s) to use during normalization. The passed value(s)
            will be broadcast to the shape of the kept axes above;
            if the value(s) cannot be broadcast, an error will be raised when
            this layer's `build()` method is called.
            `mean` and `variance` must be specified together.
        variance: The variance value(s) to use during normalization. The passed
            value(s) will be broadcast to the shape of the kept axes above;
            if the value(s) cannot be broadcast, an error will be raised when
            this layer's `build()` method is called.
            `mean` and `variance` must be specified together.
        invert: If `True`, this layer will apply the inverse transformation
            to its inputs: it would turn a normalized input back into its
            original form.

    Examples:

    Calculate a global mean and variance by analyzing the dataset in `adapt()`.

    >>> adapt_data = np.array([1., 2., 3., 4., 5.], dtype='float32')
    >>> input_data = np.array([1., 2., 3.], dtype='float32')
    >>> layer = keras.layers.Normalization(axis=None)
    >>> layer.adapt(adapt_data)
    >>> layer(input_data)
    array([-1.4142135, -0.70710677, 0.], dtype=float32)

    Calculate a mean and variance for each index on the last axis.

    >>> adapt_data = np.array([[0., 7., 4.],
    ...                        [2., 9., 6.],
    ...                        [0., 7., 4.],
    ...                        [2., 9., 6.]], dtype='float32')
    >>> input_data = np.array([[0., 7., 4.]], dtype='float32')
    >>> layer = keras.layers.Normalization(axis=-1)
    >>> layer.adapt(adapt_data)
    >>> layer(input_data)
    array([-1., -1., -1.], dtype=float32)

    Pass the mean and variance directly.

    >>> input_data = np.array([[1.], [2.], [3.]], dtype='float32')
    >>> layer = keras.layers.Normalization(mean=3., variance=2.)
    >>> layer(input_data)
    array([[-1.4142135 ],
           [-0.70710677],
           [ 0.        ]], dtype=float32)

    Use the layer to de-normalize inputs (after adapting the layer).

    >>> adapt_data = np.array([[0., 7., 4.],
    ...                        [2., 9., 6.],
    ...                        [0., 7., 4.],
    ...                        [2., 9., 6.]], dtype='float32')
    >>> input_data = np.array([[1., 2., 3.]], dtype='float32')
    >>> layer = keras.layers.Normalization(axis=-1, invert=True)
    >>> layer.adapt(adapt_data)
    >>> layer(input_data)
    array([2., 10., 8.], dtype=float32)
    """

    def __init__(
        self, axis=-1, mean=None, variance=None, invert=False, **kwargs
    ):
        super().__init__(**kwargs)
        # Standardize `axis` to a tuple.
        if axis is None:
            axis = ()
        elif isinstance(axis, int):
            axis = (axis,)
        else:
            axis = tuple(axis)
        self.axis = axis

        self.input_mean = mean
        self.input_variance = variance
        self.invert = invert
        self.supports_masking = True
        self._build_input_shape = None
        self.mean = None

        # Set `mean` and `variance` if passed.
        if (mean is not None) != (variance is not None):
            raise ValueError(
                "When setting values directly, both `mean` and `variance` "
                f"must be set. Received: mean={mean} and variance={variance}"
            )

    def build(self, input_shape):
        if input_shape is None:
            return

        ndim = len(input_shape)
        self._build_input_shape = input_shape

        if any(a < -ndim or a >= ndim for a in self.axis):
            raise ValueError(
                "All `axis` values must be in the range [-ndim, ndim). "
                f"Received inputs with ndim={ndim}, while axis={self.axis}"
            )

        # Axes to be kept, replacing negative values with positive equivalents.
        # Sorted to avoid transposing axes.
        self._keep_axis = tuple(
            sorted([d if d >= 0 else d + ndim for d in self.axis])
        )
        # All axes to be kept should have known shape.
        for d in self._keep_axis:
            if input_shape[d] is None:
                raise ValueError(
                    "All `axis` values to be kept must have a known shape. "
                    f"Received axis={self.axis}, "
                    f"inputs.shape={input_shape}, "
                    f"with unknown axis at index {d}"
                )
        # Axes to be reduced.
        self._reduce_axis = tuple(
            d for d in range(ndim) if d not in self._keep_axis
        )
        # 1 if an axis should be reduced, 0 otherwise.
        self._reduce_axis_mask = [
            0 if d in self._keep_axis else 1 for d in range(ndim)
        ]
        # Broadcast any reduced axes.
        self._broadcast_shape = [
            input_shape[d] if d in self._keep_axis else 1 for d in range(ndim)
        ]
        mean_and_var_shape = tuple(input_shape[d] for d in self._keep_axis)
        self._mean_and_var_shape = mean_and_var_shape

        if self.input_mean is None:
            self.adapt_mean = self.add_weight(
                name="mean",
                shape=mean_and_var_shape,
                initializer="zeros",
                trainable=False,
            )
            self.adapt_variance = self.add_weight(
                name="variance",
                shape=mean_and_var_shape,
                initializer="ones",
                trainable=False,
            )
            # For backwards compatibility with older saved models.
            self.count = self.add_weight(
                name="count",
                shape=(),
                dtype="int",
                initializer="zeros",
                trainable=False,
            )
            self.built = True
            self.finalize_state()
        else:
            # In the no adapt case, make constant tensors for mean and variance
            # with proper broadcast shape for use during call.
            mean = ops.convert_to_tensor(self.input_mean)
            variance = ops.convert_to_tensor(self.input_variance)
            mean = ops.broadcast_to(mean, self._broadcast_shape)
            variance = ops.broadcast_to(variance, self._broadcast_shape)
            self.mean = ops.cast(mean, dtype=self.compute_dtype)
            self.variance = ops.cast(variance, dtype=self.compute_dtype)

    def adapt(self, data):
        """Computes the mean and variance of values in a dataset.

        Calling `adapt()` on a `Normalization` layer is an alternative to
        passing in `mean` and `variance` arguments during layer construction. A
        `Normalization` layer should always either be adapted over a dataset or
        passed `mean` and `variance`.

        During `adapt()`, the layer will compute a `mean` and `variance`
        separately for each position in each axis specified by the `axis`
        argument. To calculate a single `mean` and `variance` over the input
        data, simply pass `axis=None` to the layer.

        Arg:
            data: The data to train on. It can be passed either as a
                `tf.data.Dataset`, as a NumPy array, or as a backend-native
                eager tensor.
                If a dataset, *it must be batched*. Keras will assume that the
                data is batched, and if that assumption doesn't hold, the mean
                and variance may be incorrectly computed.
        """
        if isinstance(data, np.ndarray) or backend.is_tensor(data):
            input_shape = data.shape
        elif isinstance(data, tf.data.Dataset):
            input_shape = tuple(data.element_spec.shape)
            if len(input_shape) == 1:
                # Batch dataset if it isn't batched
                data = data.batch(128)
            input_shape = tuple(data.element_spec.shape)

        if not self.built:
            self.build(input_shape)
        else:
            for d in self._keep_axis:
                if input_shape[d] != self._build_input_shape[d]:
                    raise ValueError(
                        "The layer was built with "
                        f"input_shape={self._build_input_shape}, "
                        "but adapt() is being called with data with "
                        f"an incompatible shape, data.shape={input_shape}"
                    )

        if isinstance(data, np.ndarray):
            total_mean = np.mean(data, axis=self._reduce_axis)
            total_var = np.var(data, axis=self._reduce_axis)
        elif backend.is_tensor(data):
            total_mean = ops.mean(data, axis=self._reduce_axis)
            total_var = ops.var(data, axis=self._reduce_axis)
        elif isinstance(data, tf.data.Dataset):
            total_mean = ops.zeros(self._mean_and_var_shape)
            total_var = ops.zeros(self._mean_and_var_shape)
            total_count = 0
            for batch in data:
                batch = backend.convert_to_tensor(
                    batch, dtype=self.compute_dtype
                )
                batch_mean = ops.mean(batch, axis=self._reduce_axis)
                batch_var = ops.var(batch, axis=self._reduce_axis)
                if self._reduce_axis:
                    batch_reduce_shape = (
                        batch.shape[d] for d in self._reduce_axis
                    )
                    batch_count = math.prod(batch_reduce_shape)
                else:
                    batch_count = 1

                total_count += batch_count
                batch_weight = float(batch_count) / total_count
                existing_weight = 1.0 - batch_weight
                new_total_mean = (
                    total_mean * existing_weight + batch_mean * batch_weight
                )
                # The variance is computed using the lack-of-fit sum of squares
                # formula (see
                # https://en.wikipedia.org/wiki/Lack-of-fit_sum_of_squares).
                total_var = (
                    total_var + (total_mean - new_total_mean) ** 2
                ) * existing_weight + (
                    batch_var + (batch_mean - new_total_mean) ** 2
                ) * batch_weight
                total_mean = new_total_mean
        else:
            raise NotImplementedError(f"Unsupported data type: {type(data)}")

        self.adapt_mean.assign(total_mean)
        self.adapt_variance.assign(total_var)
        self.finalize_state()

    def finalize_state(self):
        if self.input_mean is not None or not self.built:
            return

        # In the adapt case, we make constant tensors for mean and variance with
        # proper broadcast shape and dtype each time `finalize_state` is called.
        self.mean = ops.reshape(self.adapt_mean, self._broadcast_shape)
        self.mean = ops.cast(self.mean, self.compute_dtype)
        self.variance = ops.reshape(self.adapt_variance, self._broadcast_shape)
        self.variance = ops.cast(self.variance, self.compute_dtype)

    def call(self, inputs):
        # This layer can be called in tf.data
        # even with another backend after it has been adapted.
        # However it must use backend-native logic for adapt().
        if self.mean is None:
            # May happen when in tf.data when mean/var was passed explicitly
            raise ValueError(
                "You must call `.build(input_shape)` "
                "on the layer before using it."
            )
        inputs = self.backend.core.convert_to_tensor(
            inputs, dtype=self.compute_dtype
        )
        # Ensure the weights are in the correct backend. Without this, it is
        # possible to cause breakage when using this layer in tf.data.
        mean = self.convert_weight(self.mean)
        variance = self.convert_weight(self.variance)
        if self.invert:
            return self.backend.numpy.add(
                mean,
                self.backend.numpy.multiply(
                    inputs,
                    self.backend.numpy.maximum(
                        self.backend.numpy.sqrt(variance), backend.epsilon()
                    ),
                ),
            )
        else:
            return self.backend.numpy.divide(
                self.backend.numpy.subtract(inputs, mean),
                self.backend.numpy.maximum(
                    self.backend.numpy.sqrt(variance), backend.epsilon()
                ),
            )

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "axis": self.axis,
                "invert": self.invert,
                "mean": np.array(self.input_mean).tolist(),
                "variance": np.array(self.input_variance).tolist(),
            }
        )
        return config

    def load_own_variables(self, store):
        super().load_own_variables(store)
        # Ensure that we call finalize_state after variable loading.
        self.finalize_state()

    def get_build_config(self):
        if self._build_input_shape:
            return {"input_shape": self._build_input_shape}

    def build_from_config(self, config):
        if config:
            self.build(config["input_shape"])
