import numpy as np
import tensorflow as tf

from keras_core import backend
from keras_core.api_export import keras_core_export
from keras_core.layers.layer import Layer


@keras_core_export("keras_core.layers.Discretization")
class Discretization(Layer):
    """A preprocessing layer which buckets continuous features by ranges.

    This layer will place each element of its input data into one of several
    contiguous ranges and output an integer index indicating which range each
    element was placed in.

    **Note:** This layer wraps `tf.keras.layers.Discretization`. It cannot
    be used as part of the compiled computation graph of a model with
    any backend other than TensorFlow.
    It can however be used with any backend when running eagerly.
    It can also always be used as part of an input preprocessing pipeline
    with any backend (outside the model itself), which is how we recommend
    to use this layer.

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
            If this option is set,
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

    Bucketize float values based on provided buckets.
    >>> input = np.array([[-1.5, 1.0, 3.4, .5], [0.0, 3.0, 1.3, 0.0]])
    >>> layer = Discretization(bin_boundaries=[0., 1., 2.])
    >>> layer(input)
    array([[0, 2, 3, 1],
           [1, 3, 2, 1]])

    Bucketize float values based on a number of buckets to compute.
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
        name=None,
        **kwargs,
    ):
        super().__init__(name=name)
        if sparse and backend.backend() != "tensorflow":
            raise ValueError(
                "`sparse` can only be set to True with the "
                "TensorFlow backend."
            )
        self.layer = tf.keras.layers.Discretization(
            bin_boundaries=bin_boundaries,
            num_bins=num_bins,
            epsilon=epsilon,
            output_mode=output_mode,
            sparse=sparse,
            name=name,
            **kwargs,
        )
        self.bin_boundaries = (
            bin_boundaries if bin_boundaries is not None else []
        )
        self.num_bins = num_bins
        self.epsilon = epsilon
        self.output_mode = output_mode
        self.sparse = sparse

    def build(self, input_shape):
        self.layer.build(input_shape)
        self.built = True

    # We override this method solely to generate a docstring.
    def adapt(self, data, batch_size=None, steps=None):
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
            batch_size: Integer or `None`.
                Number of samples per state update.
                If unspecified, `batch_size` will default to 32.
                Do not specify the `batch_size` if your data is in the
                form of a `tf.data.Dataset`
                (it is expected to be already batched).
            steps: Integer or `None`.
                Total number of steps (batches of samples)
                When training with input tensors such as
                the default `None` is equal to
                the number of samples in your dataset divided by
                the batch size, or 1 if that cannot be determined.
                If `data` is a `tf.data.Dataset`, and `steps` is `None`,
                `adapt()` will run until the input dataset is exhausted.
                When passing an infinitely
                repeating dataset, you must specify the `steps` argument. This
                argument is not supported with array inputs or list inputs.
        """
        self.layer.adapt(data, batch_size=batch_size, steps=steps)

    def update_state(self, data):
        self.layer.update_state(data)

    def finalize_state(self):
        self.layer.finalize_state()

    def reset_state(self):
        self.layer.reset_state()

    def get_config(self):
        return {
            "bin_boundaries": self.bin_boundaries,
            "num_bins": self.num_bins,
            "epsilon": self.epsilon,
            "output_mode": self.output_mode,
            "sparse": self.sparse,
            "name": self.name,
        }

    def compute_output_shape(self, input_shape):
        return input_shape

    def call(self, inputs):
        if not isinstance(inputs, (tf.Tensor, np.ndarray)):
            inputs = tf.convert_to_tensor(np.array(inputs))
        outputs = self.layer.call(inputs)
        if backend.backend() != "tensorflow":
            outputs = backend.convert_to_tensor(outputs)
        return outputs
