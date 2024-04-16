from keras.src import backend
from keras.src.api_export import keras_export
from keras.src.layers.layer import Layer
from keras.src.utils import argument_validation
from keras.src.utils import backend_utils
from keras.src.utils import tf_utils
from keras.src.utils.module_utils import tensorflow as tf


@keras_export("keras.layers.HashedCrossing")
class HashedCrossing(Layer):
    """A preprocessing layer which crosses features using the "hashing trick".

    This layer performs crosses of categorical features using the "hashing
    trick". Conceptually, the transformation can be thought of as:
    `hash(concatenate(features)) % num_bins.

    This layer currently only performs crosses of scalar inputs and batches of
    scalar inputs. Valid input shapes are `(batch_size, 1)`, `(batch_size,)` and
    `()`.

    **Note:** This layer wraps `tf.keras.layers.HashedCrossing`. It cannot
    be used as part of the compiled computation graph of a model with
    any backend other than TensorFlow.
    It can however be used with any backend when running eagerly.
    It can also always be used as part of an input preprocessing pipeline
    with any backend (outside the model itself), which is how we recommend
    to use this layer.

    **Note:** This layer is safe to use inside a `tf.data` pipeline
    (independently of which backend you're using).

    Args:
        num_bins: Number of hash bins.
        output_mode: Specification for the output of the layer. Values can be
            `"int"`, or `"one_hot"` configuring the layer as follows:
            - `"int"`: Return the integer bin indices directly.
            - `"one_hot"`: Encodes each individual element in the input into an
                array the same size as `num_bins`, containing a 1 at the input's
                bin index. Defaults to `"int"`.
        sparse: Boolean. Only applicable to `"one_hot"` mode and only valid
            when using the TensorFlow backend. If `True`, returns
            a `SparseTensor` instead of a dense `Tensor`. Defaults to `False`.
        **kwargs: Keyword arguments to construct a layer.

    Examples:

    **Crossing two scalar features.**

    >>> layer = keras.layers.HashedCrossing(
    ...     num_bins=5)
    >>> feat1 = np.array(['A', 'B', 'A', 'B', 'A'])
    >>> feat2 = np.array([101, 101, 101, 102, 102])
    >>> layer((feat1, feat2))
    array([1, 4, 1, 1, 3])

    **Crossing and one-hotting two scalar features.**

    >>> layer = keras.layers.HashedCrossing(
    ...     num_bins=5, output_mode='one_hot')
    >>> feat1 = np.array(['A', 'B', 'A', 'B', 'A'])
    >>> feat2 = np.array([101, 101, 101, 102, 102])
    >>> layer((feat1, feat2))
    array([[0., 1., 0., 0., 0.],
            [0., 0., 0., 0., 1.],
            [0., 1., 0., 0., 0.],
            [0., 1., 0., 0., 0.],
            [0., 0., 0., 1., 0.]], dtype=float32)
    """

    def __init__(
        self,
        num_bins,
        output_mode="int",
        sparse=False,
        name=None,
        dtype=None,
        **kwargs,
    ):
        if not tf.available:
            raise ImportError(
                "Layer HashedCrossing requires TensorFlow. "
                "Install it via `pip install tensorflow`."
            )

        if output_mode == "int" and dtype is None:
            dtype = "int64"

        super().__init__(name=name, dtype=dtype)
        if sparse and backend.backend() != "tensorflow":
            raise ValueError(
                "`sparse=True` can only be used with the " "TensorFlow backend."
            )

        argument_validation.validate_string_arg(
            output_mode,
            allowable_strings=("int", "one_hot"),
            caller_name=self.__class__.__name__,
            arg_name="output_mode",
        )

        self.num_bins = num_bins
        self.output_mode = output_mode
        self.sparse = sparse
        self._allow_non_tensor_positional_args = True
        self._convert_input_args = False
        self.supports_jit = False

    def compute_output_shape(self, input_shape):
        if (
            not len(input_shape) == 2
            or not isinstance(input_shape[0], tuple)
            or not isinstance(input_shape[1], tuple)
        ):
            raise ValueError(
                "Expected as input a list/tuple of 2 tensors. "
                f"Received input_shape={input_shape}"
            )
        if input_shape[0][-1] != input_shape[1][-1]:
            raise ValueError(
                "Expected the two input tensors to have identical shapes. "
                f"Received input_shape={input_shape}"
            )

        if not input_shape:
            if self.output_mode == "int":
                return ()
            return (self.num_bins,)
        if self.output_mode == "int":
            return input_shape[0]

        if self.output_mode == "one_hot" and input_shape[0][-1] != 1:
            return tuple(input_shape[0]) + (self.num_bins,)

        return tuple(input_shape[0])[:-1] + (self.num_bins,)

    def call(self, inputs):
        self._check_at_least_two_inputs(inputs)
        inputs = [tf_utils.ensure_tensor(x) for x in inputs]
        self._check_input_shape_and_type(inputs)

        # Uprank to rank 2 for the cross_hashed op.
        rank = len(inputs[0].shape)
        if rank < 2:
            inputs = [tf_utils.expand_dims(x, -1) for x in inputs]
        if rank < 1:
            inputs = [tf_utils.expand_dims(x, -1) for x in inputs]

        # Perform the cross and convert to dense
        outputs = tf.sparse.cross_hashed(inputs, self.num_bins)
        outputs = tf.sparse.to_dense(outputs)

        # Fix output shape and downrank to match input rank.
        if rank == 2:
            # tf.sparse.cross_hashed output shape will always be None on the
            # last dimension. Given our input shape restrictions, we want to
            # force shape 1 instead.
            outputs = tf.reshape(outputs, [-1, 1])
        elif rank == 1:
            outputs = tf.reshape(outputs, [-1])
        elif rank == 0:
            outputs = tf.reshape(outputs, [])

        # Encode outputs.
        outputs = tf_utils.encode_categorical_inputs(
            outputs,
            output_mode=self.output_mode,
            depth=self.num_bins,
            sparse=self.sparse,
            dtype=self.compute_dtype,
        )
        return backend_utils.convert_tf_tensor(outputs, dtype=self.dtype)

    def get_config(self):
        return {
            "num_bins": self.num_bins,
            "output_mode": self.output_mode,
            "sparse": self.sparse,
            "name": self.name,
            "dtype": self.dtype,
        }

    def _check_at_least_two_inputs(self, inputs):
        if not isinstance(inputs, (list, tuple)):
            raise ValueError(
                "`HashedCrossing` should be called on a list or tuple of "
                f"inputs. Received: inputs={inputs}"
            )
        if len(inputs) < 2:
            raise ValueError(
                "`HashedCrossing` should be called on at least two inputs. "
                f"Received: inputs={inputs}"
            )

    def _check_input_shape_and_type(self, inputs):
        first_shape = tuple(inputs[0].shape)
        rank = len(first_shape)
        if rank > 2 or (rank == 2 and first_shape[-1] != 1):
            raise ValueError(
                "All `HashedCrossing` inputs should have shape `()`, "
                "`(batch_size)` or `(batch_size, 1)`. "
                f"Received: inputs={inputs}"
            )
        if not all(tuple(x.shape) == first_shape for x in inputs[1:]):
            raise ValueError(
                "All `HashedCrossing` inputs should have equal shape. "
                f"Received: inputs={inputs}"
            )
        if any(
            isinstance(x, (tf.RaggedTensor, tf.SparseTensor)) for x in inputs
        ):
            raise ValueError(
                "All `HashedCrossing` inputs should be dense tensors. "
                f"Received: inputs={inputs}"
            )
        if not all(
            tf.as_dtype(x.dtype).is_integer or x.dtype == tf.string
            for x in inputs
        ):
            raise ValueError(
                "All `HashedCrossing` inputs should have an integer or "
                f"string dtype. Received: inputs={inputs}"
            )
