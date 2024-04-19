from keras.src.api_export import keras_export
from keras.src.backend import KerasTensor
from keras.src.layers.preprocessing.tf_data_layer import TFDataLayer
from keras.src.utils import backend_utils


@keras_export("keras.layers.CategoryEncoding")
class CategoryEncoding(TFDataLayer):
    """A preprocessing layer which encodes integer features.

    This layer provides options for condensing data into a categorical encoding
    when the total number of tokens are known in advance. It accepts integer
    values as inputs, and it outputs a dense or sparse representation of those
    inputs. For integer inputs where the total number of tokens is not known,
    use `keras.layers.IntegerLookup` instead.

    **Note:** This layer is safe to use inside a `tf.data` pipeline
    (independently of which backend you're using).

    Examples:

    **One-hot encoding data**

    >>> layer = keras.layers.CategoryEncoding(
    ...           num_tokens=4, output_mode="one_hot")
    >>> layer([3, 2, 0, 1])
    array([[0., 0., 0., 1.],
            [0., 0., 1., 0.],
            [1., 0., 0., 0.],
            [0., 1., 0., 0.]]>

    **Multi-hot encoding data**

    >>> layer = keras.layers.CategoryEncoding(
    ...           num_tokens=4, output_mode="multi_hot")
    >>> layer([[0, 1], [0, 0], [1, 2], [3, 1]])
    array([[1., 1., 0., 0.],
            [1., 0., 0., 0.],
            [0., 1., 1., 0.],
            [0., 1., 0., 1.]]>

    **Using weighted inputs in `"count"` mode**

    >>> layer = keras.layers.CategoryEncoding(
    ...           num_tokens=4, output_mode="count")
    >>> count_weights = np.array([[.1, .2], [.1, .1], [.2, .3], [.4, .2]])
    >>> layer([[0, 1], [0, 0], [1, 2], [3, 1]], count_weights=count_weights)
      array([[0.1, 0.2, 0. , 0. ],
             [0.2, 0. , 0. , 0. ],
             [0. , 0.2, 0.3, 0. ],
             [0. , 0.2, 0. , 0.4]]>

    Args:
        num_tokens: The total number of tokens the layer should support. All
            inputs to the layer must integers in the range `0 <= value <
            num_tokens`, or an error will be thrown.
        output_mode: Specification for the output of the layer.
            Values can be `"one_hot"`, `"multi_hot"` or `"count"`,
            configuring the layer as follows:
                - `"one_hot"`: Encodes each individual element in the input
                    into an array of `num_tokens` size, containing a 1 at the
                    element index. If the last dimension is size 1, will encode
                    on that dimension. If the last dimension is not size 1,
                    will append a new dimension for the encoded output.
                - `"multi_hot"`: Encodes each sample in the input into a single
                    array of `num_tokens` size, containing a 1 for each
                    vocabulary term present in the sample. Treats the last
                    dimension as the sample dimension, if input shape is
                    `(..., sample_length)`, output shape will be
                    `(..., num_tokens)`.
                - `"count"`: Like `"multi_hot"`, but the int array contains a
                    count of the number of times the token at that index
                    appeared in the sample.
            For all output modes, currently only output up to rank 2 is
            supported.
            Defaults to `"multi_hot"`.
        sparse: Whether to return a sparse tensor; for backends that support
            sparse tensors.

    Call arguments:
        inputs: A 1D or 2D tensor of integer inputs.
        count_weights: A tensor in the same shape as `inputs` indicating the
            weight for each sample value when summing up in `count` mode.
            Not used in `"multi_hot"` or `"one_hot"` modes.
    """

    def __init__(
        self, num_tokens=None, output_mode="multi_hot", sparse=False, **kwargs
    ):
        super().__init__(**kwargs)

        # Support deprecated names for output_modes.
        if output_mode == "binary":
            output_mode = "multi_hot"

        # 'output_mode' must be one of ("count", "one_hot", "multi_hot")
        if output_mode not in ("count", "one_hot", "multi_hot"):
            raise ValueError(f"Unknown arg for output_mode: {output_mode}")

        if num_tokens is None:
            raise ValueError(
                "num_tokens must be set to use this layer. If the "
                "number of tokens is not known beforehand, use the "
                "IntegerLookup layer instead."
            )
        if num_tokens < 1:
            raise ValueError(
                f"`num_tokens` must be >= 1. Received: num_tokens={num_tokens}."
            )
        self.num_tokens = num_tokens
        self.output_mode = output_mode
        self.sparse = sparse
        self._allow_non_tensor_positional_args = True
        self._convert_input_args = False

    def _count(self, inputs, axis=-1, count_weights=None):
        # We don't use `ops.bincount` here because its output has a dynamic
        # shape (the last dimension is calculated from the highest value of
        # `inputs`). Instead, we reimplement a narrower use case where
        # `minlength` and `maxlength` (not supported by `ops.bincount`) are a
        # static value and the same value of `self.num_tokens`. Also, we don't
        # need to support indices that are negative or greater than
        # `self.num_tokens`.
        reduction_axis = 1 if len(inputs.shape) > 1 else 0

        one_hot_encoding = self.backend.nn.one_hot(
            inputs,
            self.num_tokens,
            axis=axis,
            dtype=self.dtype,
            sparse=self.sparse,
        )
        if count_weights is not None:
            split_weights = self.backend.numpy.split(
                count_weights,
                count_weights.shape[reduction_axis],
                reduction_axis,
            )
            stacked_weights = self.backend.numpy.stack(
                split_weights, axis=reduction_axis
            )
            one_hot_encoding = one_hot_encoding * stacked_weights

        outputs = self.backend.numpy.sum(
            one_hot_encoding,
            axis=reduction_axis,
        )
        return outputs

    def _encode(self, inputs, count_weights=None):
        if self.output_mode == "multi_hot":
            return self.backend.nn.multi_hot(
                inputs, self.num_tokens, dtype=self.dtype, sparse=self.sparse
            )
        elif self.output_mode == "one_hot":
            return self.backend.nn.one_hot(
                inputs, self.num_tokens, dtype=self.dtype, sparse=self.sparse
            )
        elif self.output_mode == "count":
            return self._count(inputs, count_weights=count_weights)

    def compute_output_shape(self, input_shape):
        # Treat rank-0 inputs inconsistent with actual output .
        if (input_shape is not None) & (len(input_shape) == 0):
            # If output_mode == "multi_hot" return same shape .
            if self.output_mode == "multi_hot":
                return input_shape
            # If output_mode == "one_hot" append num_tokens to shape .
            # This is inline with actual output .
            elif self.output_mode == "one_hot":
                return (self.num_tokens,)
        if self.output_mode == "one_hot":
            if input_shape[-1] != 1:
                return tuple(input_shape + (self.num_tokens,))
            elif len(input_shape) == 1:
                return tuple(input_shape + (self.num_tokens,))
            else:
                return tuple(input_shape[:-1] + (self.num_tokens,))
        return tuple(input_shape[:-1] + (self.num_tokens,))

    def compute_output_spec(self, inputs, count_weights=None):
        output_shape = self.compute_output_shape(inputs.shape)
        return KerasTensor(
            output_shape, dtype=self.compute_dtype, sparse=self.sparse
        )

    def get_config(self):
        config = {
            "num_tokens": self.num_tokens,
            "output_mode": self.output_mode,
        }
        base_config = super().get_config()
        return {**base_config, **config}

    def call(self, inputs, count_weights=None):
        if count_weights is not None:
            if self.output_mode != "count":
                raise ValueError(
                    "`count_weights` is not used when `output_mode` is not "
                    "`'count'`. Received `count_weights={count_weights}`."
                )
            count_weights = self.backend.convert_to_tensor(
                count_weights, dtype=self.compute_dtype
            )
        outputs = self._encode(inputs, count_weights)
        return backend_utils.convert_tf_tensor(outputs)
