import numpy as np

from keras.src import backend
from keras.src.api_export import keras_export
from keras.src.layers.preprocessing.tf_data_layer import TFDataLayer
from keras.src.utils import argument_validation
from keras.src.utils import numerical_utils
from keras.src.utils.module_utils import tensorflow as tf


@keras_export("keras.layers.Discretization")
class Discretization(TFDataLayer):
    """A preprocessing layer which buckets continuous features by ranges.

    This layer will place each element of its input data into one of several
    contiguous ranges and output an integer index indicating which range each
    element was placed in.

    **Note:** This layer is safe to use inside a `tf.data` pipeline
    (independently of which backend you're using).

    Input shape:
        Any array of dimension 2 or higher.

    Output shape:
        Same as input shape.

    Arguments:
        bin_boundaries: A list of bin boundaries.
            The leftmost and rightmost bins
            will always extend to `-inf` and `inf`,
            so `bin_boundaries=[0., 1., 2.]`
            generates bins `(-inf, 0.)`, `[0., 1.)`, `[1., 2.)`,
            and `[2., +inf)`.
            If this option is set, `adapt()` should not be called.
        num_bins: The integer number of bins to compute.
            If this option is set, `bin_boundaries` should not be set and
            `adapt()` should be called to learn the bin boundaries.
        epsilon: Error tolerance, typically a small fraction
            close to zero (e.g. 0.01). Higher values of epsilon increase
            the quantile approximation, and hence result in more
            unequal buckets, but could improve performance
            and resource consumption.
        output_mode: Specification for the output of the layer.
            Values can be `"int"`, `"one_hot"`, `"multi_hot"`, or
            `"count"` configuring the layer as follows:
            - `"int"`: Return the discretized bin indices directly.
            - `"one_hot"`: Encodes each individual element in the
                input into an array the same size as `num_bins`,
                containing a 1 at the input's bin
                index. If the last dimension is size 1, will encode on that
                dimension.  If the last dimension is not size 1,
                will append a new dimension for the encoded output.
            - `"multi_hot"`: Encodes each sample in the input into a
                single array the same size as `num_bins`,
                containing a 1 for each bin index
                index present in the sample.
                Treats the last dimension as the sample
                dimension, if input shape is `(..., sample_length)`,
                output shape will be `(..., num_tokens)`.
            - `"count"`: As `"multi_hot"`, but the int array contains
                a count of the number of times the bin index appeared
                in the sample.
            Defaults to `"int"`.
        sparse: Boolean. Only applicable to `"one_hot"`, `"multi_hot"`,
            and `"count"` output modes. Only supported with TensorFlow
            backend. If `True`, returns a `SparseTensor` instead of
            a dense `Tensor`. Defaults to `False`.

    Examples:

    Discretize float values based on provided buckets.
    >>> input = np.array([[-1.5, 1.0, 3.4, .5], [0.0, 3.0, 1.3, 0.0]])
    >>> layer = Discretization(bin_boundaries=[0., 1., 2.])
    >>> layer(input)
    array([[0, 2, 3, 1],
           [1, 3, 2, 1]])

    Discretize float values based on a number of buckets to compute.
    >>> input = np.array([[-1.5, 1.0, 3.4, .5], [0.0, 3.0, 1.3, 0.0]])
    >>> layer = Discretization(num_bins=4, epsilon=0.01)
    >>> layer.adapt(input)
    >>> layer(input)
    array([[0, 2, 3, 2],
           [1, 3, 3, 1]])
    """

    def __init__(
        self,
        bin_boundaries=None,
        num_bins=None,
        epsilon=0.01,
        output_mode="int",
        sparse=False,
        dtype=None,
        name=None,
    ):
        if dtype is None:
            dtype = "int64" if output_mode == "int" else backend.floatx()

        super().__init__(name=name, dtype=dtype)

        if sparse and not backend.SUPPORTS_SPARSE_TENSORS:
            raise ValueError(
                f"`sparse=True` cannot be used with backend {backend.backend()}"
            )
        if sparse and output_mode == "int":
            raise ValueError(
                "`sparse=True` may only be used if `output_mode` is "
                "`'one_hot'`, `'multi_hot'`, or `'count'`. "
                f"Received: sparse={sparse} and "
                f"output_mode={output_mode}"
            )

        argument_validation.validate_string_arg(
            output_mode,
            allowable_strings=(
                "int",
                "one_hot",
                "multi_hot",
                "count",
            ),
            caller_name=self.__class__.__name__,
            arg_name="output_mode",
        )

        if num_bins is not None and num_bins < 0:
            raise ValueError(
                "`num_bins` must be greater than or equal to 0. "
                f"Received: `num_bins={num_bins}`"
            )
        if num_bins is not None and bin_boundaries is not None:
            raise ValueError(
                "Both `num_bins` and `bin_boundaries` should not be set. "
                f"Received: `num_bins={num_bins}` and "
                f"`bin_boundaries={bin_boundaries}`"
            )
        if num_bins is None and bin_boundaries is None:
            raise ValueError(
                "You need to set either `num_bins` or `bin_boundaries`."
            )

        self.bin_boundaries = bin_boundaries
        self.num_bins = num_bins
        self.epsilon = epsilon
        self.output_mode = output_mode
        self.sparse = sparse

        if self.bin_boundaries:
            self.summary = None
        else:
            self.summary = np.array([[], []], dtype="float32")

    def build(self, input_shape=None):
        self.built = True

    @property
    def input_dtype(self):
        return backend.floatx()

    def adapt(self, data, steps=None):
        """Computes bin boundaries from quantiles in a input dataset.

        Calling `adapt()` on a `Discretization` layer is an alternative to
        passing in a `bin_boundaries` argument during construction. A
        `Discretization` layer should always be either adapted over a dataset or
        passed `bin_boundaries`.

        During `adapt()`, the layer will estimate the quantile boundaries of the
        input dataset. The number of quantiles can be controlled via the
        `num_bins` argument, and the error tolerance for quantile boundaries can
        be controlled via the `epsilon` argument.

        Arguments:
            data: The data to train on. It can be passed either as a
                batched `tf.data.Dataset`,
                or as a NumPy array.
            steps: Integer or `None`.
                Total number of steps (batches of samples) to process.
                If `data` is a `tf.data.Dataset`, and `steps` is `None`,
                `adapt()` will run until the input dataset is exhausted.
                When passing an infinitely
                repeating dataset, you must specify the `steps` argument. This
                argument is not supported with array inputs or list inputs.
        """
        if self.num_bins is None:
            raise ValueError(
                "Cannot adapt a Discretization layer that has been initialized "
                "with `bin_boundaries`, use `num_bins` instead."
            )
        self.reset_state()
        if isinstance(data, tf.data.Dataset):
            if steps is not None:
                data = data.take(steps)
            for batch in data:
                self.update_state(batch)
        else:
            self.update_state(data)
        self.finalize_state()

    def update_state(self, data):
        data = np.array(data).astype("float32")
        summary = summarize(data, self.epsilon)
        self.summary = merge_summaries(summary, self.summary, self.epsilon)

    def finalize_state(self):
        if self.num_bins is None:
            return
        self.bin_boundaries = get_bin_boundaries(
            self.summary, self.num_bins
        ).tolist()

    def reset_state(self):
        if self.num_bins is None:
            return
        self.summary = np.array([[], []], dtype="float32")

    def compute_output_spec(self, inputs):
        return backend.KerasTensor(shape=inputs.shape, dtype=self.compute_dtype)

    def load_own_variables(self, store):
        if len(store) == 1:
            # Legacy format case
            self.summary = store["0"]
        return

    def call(self, inputs):
        if self.bin_boundaries is None:
            raise ValueError(
                "You need to either pass the `bin_boundaries` argument at "
                "construction time or call `adapt(dataset)` before you can "
                "start using the `Discretization` layer."
            )

        indices = self.backend.numpy.digitize(inputs, self.bin_boundaries)
        return numerical_utils.encode_categorical_inputs(
            indices,
            output_mode=self.output_mode,
            depth=len(self.bin_boundaries) + 1,
            dtype=self.compute_dtype,
            sparse=self.sparse,
            backend_module=self.backend,
        )

    def get_config(self):
        return {
            "bin_boundaries": self.bin_boundaries,
            "num_bins": self.num_bins,
            "epsilon": self.epsilon,
            "output_mode": self.output_mode,
            "sparse": self.sparse,
            "name": self.name,
            "dtype": self.dtype,
        }

    @classmethod
    def from_config(cls, config, custom_objects=None):
        if (
            config.get("bin_boundaries", None) is not None
            and config.get("num_bins", None) is not None
        ):
            # After `adapt` was called, both `bin_boundaries` and `num_bins` are
            # populated, but `__init__` won't let us create a new layer with
            # both `bin_boundaries` and `num_bins`. We therefore apply
            # `bin_boundaries` after creation.
            config = config.copy()
            bin_boundaries = config.pop("bin_boundaries")
            discretization = cls(**config)
            discretization.bin_boundaries = bin_boundaries
            return discretization
        return cls(**config)


def summarize(values, epsilon):
    """Reduce a 1D sequence of values to a summary.

    This algorithm is based on numpy.quantiles but modified to allow for
    intermediate steps between multiple data sets. It first finds the target
    number of bins as the reciprocal of epsilon and then takes the individual
    values spaced at appropriate intervals to arrive at that target.
    The final step is to return the corresponding counts between those values
    If the target num_bins is larger than the size of values, the whole array is
    returned (with weights of 1).

    Args:
        values: 1D `np.ndarray` to be summarized.
        epsilon: A `'float32'` that determines the approximate desired
        precision.

    Returns:
        A 2D `np.ndarray` that is a summary of the inputs. First column is the
        interpolated partition values, the second is the weights (counts).
    """
    values = np.reshape(values, [-1])
    values = np.sort(values)
    elements = np.size(values)
    num_buckets = 1.0 / epsilon
    increment = elements / num_buckets
    start = increment
    step = max(increment, 1)
    boundaries = values[int(start) :: int(step)]
    weights = np.ones_like(boundaries)
    weights = weights * step
    return np.stack([boundaries, weights])


def merge_summaries(prev_summary, next_summary, epsilon):
    """Weighted merge sort of summaries.

    Given two summaries of distinct data, this function merges (and compresses)
    them to stay within `epsilon` error tolerance.

    Args:
        prev_summary: 2D `np.ndarray` summary to be merged with `next_summary`.
        next_summary: 2D `np.ndarray` summary to be merged with `prev_summary`.
        epsilon: A float that determines the approximate desired precision.

    Returns:
        A 2-D `np.ndarray` that is a merged summary. First column is the
        interpolated partition values, the second is the weights (counts).
    """
    merged = np.concatenate((prev_summary, next_summary), axis=1)
    merged = np.take(merged, np.argsort(merged[0]), axis=1)
    return compress_summary(merged, epsilon)


def get_bin_boundaries(summary, num_bins):
    return compress_summary(summary, 1.0 / num_bins)[0, :-1]


def compress_summary(summary, epsilon):
    """Compress a summary to within `epsilon` accuracy.

    The compression step is needed to keep the summary sizes small after
    merging, and also used to return the final target boundaries. It finds the
    new bins based on interpolating cumulative weight percentages from the large
    summary.  Taking the difference of the cumulative weights from the previous
    bin's cumulative weight will give the new weight for that bin.

    Args:
        summary: 2D `np.ndarray` summary to be compressed.
        epsilon: A `'float32'` that determines the approximate desired
        precision.

    Returns:
        A 2D `np.ndarray` that is a compressed summary. First column is the
        interpolated partition values, the second is the weights (counts).
    """
    if summary.shape[1] * epsilon < 1:
        return summary

    percents = epsilon + np.arange(0.0, 1.0, epsilon)
    cum_weights = summary[1].cumsum()
    cum_weight_percents = cum_weights / cum_weights[-1]
    new_bins = np.interp(percents, cum_weight_percents, summary[0])
    cum_weights = np.interp(percents, cum_weight_percents, cum_weights)
    new_weights = cum_weights - np.concatenate(
        (np.array([0]), cum_weights[:-1])
    )
    summary = np.stack((new_bins, new_weights))
    return summary.astype("float32")
