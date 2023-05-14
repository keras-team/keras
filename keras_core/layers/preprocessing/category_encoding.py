from keras_core import operations as ops
from keras_core.api_export import keras_core_export
from keras_core.layers.layer import Layer


@keras_core_export("keras_core.layers.CategoryEncoding")
class CategoryEncoding(Layer):
    """A preprocessing layer which encodes integer features.

    This layer provides options for condensing data into a categorical encoding
    when the total number of tokens are known in advance. It accepts integer
    values as inputs, and it outputs a dense or sparse representation of those
    inputs. For integer inputs where the total number of tokens is not known,
    use `keras_core.layers.IntegerLookup` instead.

    Examples:

    **One-hot encoding data**

    >>> layer = keras_core.layers.CategoryEncoding(
    ...           num_tokens=4, output_mode="one_hot")
    >>> layer([3, 2, 0, 1])
    array([[0., 0., 0., 1.],
            [0., 0., 1., 0.],
            [1., 0., 0., 0.],
            [0., 1., 0., 0.]]>

    **Multi-hot encoding data**

    >>> layer = keras_core.layers.CategoryEncoding(
    ...           num_tokens=4, output_mode="multi_hot")
    >>> layer([[0, 1], [0, 0], [1, 2], [3, 1]])
    array([[1., 1., 0., 0.],
            [1., 0., 0., 0.],
            [0., 1., 1., 0.],
            [0., 1., 0., 1.]]>

    **Using weighted inputs in `"count"` mode**

    >>> layer = keras_core.layers.CategoryEncoding(
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

    Call arguments:
        inputs: A 1D or 2D tensor of integer inputs.
        count_weights: A tensor in the same shape as `inputs` indicating the
            weight for each sample value when summing up in `count` mode.
            Not used in `"multi_hot"` or `"one_hot"` modes.
    """

    def __init__(self, num_tokens=None, output_mode="multi_hot", **kwargs):
        # max_tokens is an old name for the num_tokens arg we continue to
        # support because of usage.
        if "max_tokens" in kwargs:
            num_tokens = kwargs["max_tokens"]
            del kwargs["max_tokens"]

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

    def compute_output_shape(self, input_shape):
        input_shape = list(input_shape)
        if not input_shape:
            return ops.shape(self.num_tokens)
        if self.output_mode == "one_hot" and input_shape[-1] != 1:
            return tuple(input_shape + [self.num_tokens])
        else:
            return tuple(input_shape[:-1] + [self.num_tokens])

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
            count_weights = ops.cast(count_weights, self.compute_dtype)

        depth = self.num_tokens

        max_value = ops.amax(inputs)
        min_value = ops.amin(inputs)
        condition = ops.logical_and(
            ops.greater(ops.cast(depth, max_value.dtype), max_value),
            ops.greater_equal(min_value, ops.cast(0, min_value.dtype)),
        )
        if not condition:
            raise ValueError(
                "Input values must be in the range 0 <= values < num_tokens"
                f" with num_tokens={depth}"
            )

        return self._encode_categorical_inputs(
            inputs,
            output_mode=self.output_mode,
            depth=depth,
            count_weights=count_weights,
        )

    def _encode_categorical_inputs(
        self,
        inputs,
        output_mode,
        depth,
        count_weights=None,
    ):
        # In all cases, we should uprank scalar input to a single sample.
        if len(inputs.shape) == 0:
            inputs = ops.expand_dims(inputs, -1)
        # One hot will uprank only if the final output dimension
        # is not already 1.
        if output_mode == "one_hot":
            if len(inputs.shape) > 1 and inputs.shape[-1] != 1:
                inputs = ops.expand_dims(inputs, -1)

        # TODO(b/190445202): remove output rank restriction.
        if len(inputs.shape) > 2:
            raise ValueError(
                "When output_mode is not `'int'`, maximum supported "
                f"output rank is 2. Received output_mode {output_mode} "
                f"and input shape {inputs.shape}, "
                f"which would result in output rank {len(inputs.shape)}."
            )

        binary_output = output_mode in ("multi_hot", "one_hot")
        inputs = ops.cast(inputs, "int32")

        if binary_output:
            bincounts = ops.one_hot(inputs, num_classes=depth)
            if output_mode == "multi_hot":
                bincounts = ops.sum(bincounts, axis=0)
        else:
            bincounts = ops.bincount(
                inputs, minlength=depth, weights=count_weights
            )

        return bincounts
