from keras_core import backend
from keras_core.api_export import keras_core_export
from keras_core.layers.layer import Layer
from keras_core.utils import backend_utils


@keras_core_export("keras_core.layers.CategoryEncoding")
class CategoryEncoding(Layer):
    """A preprocessing layer which encodes integer features.

    This layer provides options for condensing data into a categorical encoding
    when the total number of tokens are known in advance. It accepts integer
    values as inputs, and it outputs a dense or sparse representation of those
    inputs. For integer inputs where the total number of tokens is not known,
    use `keras_core.layers.IntegerLookup` instead.

    **Note:** This layer is safe to use inside a `tf.data` pipeline
    (independently of which backend you're using).

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
        try:
            import tensorflow as tf
        except ImportError:
            raise ImportError(
                "Layer CategoryEncoding requires TensorFlow. "
                "Install it via `pip install tensorflow`."
            )

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

        self.layer = tf.keras.layers.CategoryEncoding(
            num_tokens=num_tokens,
            output_mode=output_mode,
            **kwargs,
        )
        self._allow_non_tensor_positional_args = True
        self._convert_input_args = False

    def compute_output_shape(self, input_shape):
        return tuple(self.layer.compute_output_shape(input_shape))

    def get_config(self):
        config = {
            "num_tokens": self.num_tokens,
            "output_mode": self.output_mode,
        }
        base_config = super().get_config()
        return {**base_config, **config}

    def call(self, inputs):
        outputs = self.layer.call(inputs)
        if (
            backend.backend() != "tensorflow"
            and not backend_utils.in_tf_graph()
        ):
            outputs = backend.convert_to_tensor(outputs)
        return outputs
