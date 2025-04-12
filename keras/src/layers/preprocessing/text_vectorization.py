import numpy as np

from keras.src import backend
from keras.src.api_export import keras_export
from keras.src.layers.layer import Layer
from keras.src.layers.preprocessing.index_lookup import listify_tensors
from keras.src.layers.preprocessing.string_lookup import StringLookup
from keras.src.saving import serialization_lib
from keras.src.utils import argument_validation
from keras.src.utils import backend_utils
from keras.src.utils import tf_utils
from keras.src.utils.module_utils import tensorflow as tf


@keras_export("keras.layers.TextVectorization")
class TextVectorization(Layer):
    """A preprocessing layer which maps text features to integer sequences.

    This layer has basic options for managing text in a Keras model. It
    transforms a batch of strings (one example = one string) into either a list
    of token indices (one example = 1D tensor of integer token indices) or a
    dense representation (one example = 1D tensor of float values representing
    data about the example's tokens). This layer is meant to handle natural
    language inputs. To handle simple string inputs (categorical strings or
    pre-tokenized strings) see `kers_core.layers.StringLookup`.

    The vocabulary for the layer must be either supplied on construction or
    learned via `adapt()`. When this layer is adapted, it will analyze the
    dataset, determine the frequency of individual string values, and create a
    vocabulary from them. This vocabulary can have unlimited size or be capped,
    depending on the configuration options for this layer; if there are more
    unique values in the input than the maximum vocabulary size, the most
    frequent terms will be used to create the vocabulary.

    The processing of each example contains the following steps:

    1. Standardize each example (usually lowercasing + punctuation stripping)
    2. Split each example into substrings (usually words)
    3. Recombine substrings into tokens (usually ngrams)
    4. Index tokens (associate a unique int value with each token)
    5. Transform each example using this index, either into a vector of ints or
       a dense float vector.

    Some notes on passing callables to customize splitting and normalization for
    this layer:

    1. Any callable can be passed to this Layer, but if you want to serialize
       this object you should only pass functions that are registered Keras
       serializables (see `keras.saving.register_keras_serializable`
       for more details).
    2. When using a custom callable for `standardize`, the data received
       by the callable will be exactly as passed to this layer. The callable
       should return a tensor of the same shape as the input.
    3. When using a custom callable for `split`, the data received by the
       callable will have the 1st dimension squeezed out - instead of
       `[["string to split"], ["another string to split"]]`, the Callable will
       see `["string to split", "another string to split"]`.
       The callable should return a `tf.Tensor` of dtype `string`
       with the first dimension containing the split tokens -
       in this example, we should see something like `[["string", "to",
       "split"], ["another", "string", "to", "split"]]`.

    **Note:** This layer uses TensorFlow internally. It cannot
    be used as part of the compiled computation graph of a model with
    any backend other than TensorFlow.
    It can however be used with any backend when running eagerly.
    It can also always be used as part of an input preprocessing pipeline
    with any backend (outside the model itself), which is how we recommend
    to use this layer.

    **Note:** This layer is safe to use inside a `tf.data` pipeline
    (independently of which backend you're using).

    Args:
        max_tokens: Maximum size of the vocabulary for this layer. This should
            only be specified when adapting a vocabulary or when setting
            `pad_to_max_tokens=True`. Note that this vocabulary
            contains 1 OOV token, so the effective number of tokens is
            `(max_tokens - 1 - (1 if output_mode == "int" else 0))`.
        standardize: Optional specification for standardization to apply to the
            input text. Values can be:
            - `None`: No standardization.
            - `"lower_and_strip_punctuation"`: Text will be lowercased and all
                punctuation removed.
            - `"lower"`: Text will be lowercased.
            - `"strip_punctuation"`: All punctuation will be removed.
            - Callable: Inputs will passed to the callable function,
                which should be standardized and returned.
        split: Optional specification for splitting the input text.
            Values can be:
            - `None`: No splitting.
            - `"whitespace"`: Split on whitespace.
            - `"character"`: Split on each unicode character.
            - Callable: Standardized inputs will passed to the callable
                function, which should be split and returned.
        ngrams: Optional specification for ngrams to create from the
            possibly-split input text. Values can be `None`, an integer
            or tuple of integers; passing an integer will create ngrams
            up to that integer, and passing a tuple of integers will
            create ngrams for the specified values in the tuple.
            Passing `None` means that no ngrams will be created.
        output_mode: Optional specification for the output of the layer.
            Values can be `"int"`, `"multi_hot"`, `"count"` or `"tf_idf"`,
            configuring the layer as follows:
            - `"int"`: Outputs integer indices, one integer index per split
                string token. When `output_mode == "int"`,
                0 is reserved for masked locations;
                this reduces the vocab size to `max_tokens - 2`
                instead of `max_tokens - 1`.
            - `"multi_hot"`: Outputs a single int array per batch, of either
                vocab_size or max_tokens size, containing 1s in all elements
                where the token mapped to that index exists at least
                once in the batch item.
            - `"count"`: Like `"multi_hot"`, but the int array contains
                a count of the number of times the token at that index
                appeared in the batch item.
            - `"tf_idf"`: Like `"multi_hot"`, but the TF-IDF algorithm
                is applied to find the value in each token slot.
            For `"int"` output, any shape of input and output is supported.
            For all other output modes, currently only rank 1 inputs
            (and rank 2 outputs after splitting) are supported.
        output_sequence_length: Only valid in INT mode. If set, the output will
            have its time dimension padded or truncated to exactly
            `output_sequence_length` values, resulting in a tensor of shape
            `(batch_size, output_sequence_length)` regardless of how many tokens
            resulted from the splitting step. Defaults to `None`. If `ragged`
            is `True` then `output_sequence_length` may still truncate the
            output.
        pad_to_max_tokens: Only valid in  `"multi_hot"`, `"count"`,
            and `"tf_idf"` modes. If `True`, the output will have
            its feature axis padded to `max_tokens` even if the number
            of unique tokens in the vocabulary is less than `max_tokens`,
            resulting in a tensor of shape `(batch_size, max_tokens)`
            regardless of vocabulary size. Defaults to `False`.
        vocabulary: Optional. Either an array of strings or a string path to a
            text file. If passing an array, can pass a tuple, list,
            1D NumPy array, or 1D tensor containing the string vocabulary terms.
            If passing a file path, the file should contain one line per term
            in the vocabulary. If this argument is set,
            there is no need to `adapt()` the layer.
        idf_weights: Only valid when `output_mode` is `"tf_idf"`. A tuple, list,
            1D NumPy array, or 1D tensor of the same length as the vocabulary,
            containing the floating point inverse document frequency weights,
            which will be multiplied by per sample term counts for
            the final `tf_idf` weight. If the `vocabulary` argument is set,
            and `output_mode` is `"tf_idf"`, this argument must be supplied.
        ragged: Boolean. Only applicable to `"int"` output mode.
            Only supported with TensorFlow backend.
            If `True`, returns a `RaggedTensor` instead of a dense `Tensor`,
            where each sequence may have a different length
            after string splitting. Defaults to `False`.
        sparse: Boolean. Only applicable to `"multi_hot"`, `"count"`, and
            `"tf_idf"` output modes. Only supported with TensorFlow
            backend. If `True`, returns a `SparseTensor`
            instead of a dense `Tensor`. Defaults to `False`.
        encoding: Optional. The text encoding to use to interpret the input
            strings. Defaults to `"utf-8"`.

    Examples:

    This example instantiates a `TextVectorization` layer that lowercases text,
    splits on whitespace, strips punctuation, and outputs integer vocab indices.

    >>> max_tokens = 5000  # Maximum vocab size.
    >>> max_len = 4  # Sequence length to pad the outputs to.
    >>> # Create the layer.
    >>> vectorize_layer = TextVectorization(
    ...     max_tokens=max_tokens,
    ...     output_mode='int',
    ...     output_sequence_length=max_len)

    >>> # Now that the vocab layer has been created, call `adapt` on the
    >>> # list of strings to create the vocabulary.
    >>> vectorize_layer.adapt(["foo bar", "bar baz", "baz bada boom"])

    >>> # Now, the layer can map strings to integers -- you can use an
    >>> # embedding layer to map these integers to learned embeddings.
    >>> input_data = [["foo qux bar"], ["qux baz"]]
    >>> vectorize_layer(input_data)
    array([[4, 1, 3, 0],
           [1, 2, 0, 0]])

    This example instantiates a `TextVectorization` layer by passing a list
    of vocabulary terms to the layer's `__init__()` method.

    >>> vocab_data = ["earth", "wind", "and", "fire"]
    >>> max_len = 4  # Sequence length to pad the outputs to.
    >>> # Create the layer, passing the vocab directly. You can also pass the
    >>> # vocabulary arg a path to a file containing one vocabulary word per
    >>> # line.
    >>> vectorize_layer = keras.layers.TextVectorization(
    ...     max_tokens=max_tokens,
    ...     output_mode='int',
    ...     output_sequence_length=max_len,
    ...     vocabulary=vocab_data)

    >>> # Because we've passed the vocabulary directly, we don't need to adapt
    >>> # the layer - the vocabulary is already set. The vocabulary contains the
    >>> # padding token ('') and OOV token ('[UNK]')
    >>> # as well as the passed tokens.
    >>> vectorize_layer.get_vocabulary()
    ['', '[UNK]', 'earth', 'wind', 'and', 'fire']
    """

    def __init__(
        self,
        max_tokens=None,
        standardize="lower_and_strip_punctuation",
        split="whitespace",
        ngrams=None,
        output_mode="int",
        output_sequence_length=None,
        pad_to_max_tokens=False,
        vocabulary=None,
        idf_weights=None,
        sparse=False,
        ragged=False,
        encoding="utf-8",
        name=None,
        **kwargs,
    ):
        if not tf.available:
            raise ImportError(
                "Layer TextVectorization requires TensorFlow. "
                "Install it via `pip install tensorflow`."
            )
        if sparse and backend.backend() != "tensorflow":
            raise ValueError(
                "`sparse=True` can only be used with the TensorFlow backend."
            )
        if ragged and backend.backend() != "tensorflow":
            raise ValueError(
                "`ragged=True` can only be used with the TensorFlow backend."
            )

        # 'standardize' must be one of
        # (None, "lower_and_strip_punctuation", "lower", "strip_punctuation",
        # callable)
        argument_validation.validate_string_arg(
            standardize,
            allowable_strings=(
                "lower_and_strip_punctuation",
                "lower",
                "strip_punctuation",
            ),
            caller_name=self.__class__.__name__,
            arg_name="standardize",
            allow_none=True,
            allow_callables=True,
        )

        # 'split' must be one of (None, "whitespace", "character", callable)
        argument_validation.validate_string_arg(
            split,
            allowable_strings=("whitespace", "character"),
            caller_name=self.__class__.__name__,
            arg_name="split",
            allow_none=True,
            allow_callables=True,
        )

        # Support deprecated names for output_modes.
        if output_mode == "binary":
            output_mode = "multi_hot"
        if output_mode == "tf-idf":
            output_mode = "tf_idf"
        argument_validation.validate_string_arg(
            output_mode,
            allowable_strings=(
                "int",
                "one_hot",
                "multi_hot",
                "count",
                "tf_idf",
            ),
            caller_name=self.__class__.__name__,
            arg_name="output_mode",
        )

        # 'ngrams' must be one of (None, int, tuple(int))
        if not (
            ngrams is None
            or isinstance(ngrams, int)
            or isinstance(ngrams, tuple)
            and all(isinstance(item, int) for item in ngrams)
        ):
            raise ValueError(
                "`ngrams` must be None, an integer, or a tuple of "
                f"integers. Received: ngrams={ngrams}"
            )

        # 'output_sequence_length' must be one of (None, int) and is only
        # set if output_mode is "int"".
        if output_mode == "int" and not (
            isinstance(output_sequence_length, int)
            or (output_sequence_length is None)
        ):
            raise ValueError(
                "`output_sequence_length` must be either None or an "
                "integer when `output_mode` is 'int'. Received: "
                f"output_sequence_length={output_sequence_length}"
            )

        if output_mode != "int" and output_sequence_length is not None:
            raise ValueError(
                "`output_sequence_length` must not be set if `output_mode` is "
                "not 'int'. "
                f"Received output_sequence_length={output_sequence_length}."
            )

        if ragged and output_mode != "int":
            raise ValueError(
                "`ragged` must not be true if `output_mode` is "
                f"`'int'`. Received: ragged={ragged} and "
                f"output_mode={output_mode}"
            )

        self._max_tokens = max_tokens
        self._standardize = standardize
        self._split = split
        self._ngrams_arg = ngrams
        if isinstance(ngrams, int):
            self._ngrams = tuple(range(1, ngrams + 1))
        else:
            self._ngrams = ngrams
        self._ragged = ragged

        self._output_mode = output_mode
        self._output_sequence_length = output_sequence_length
        self._encoding = encoding

        # We save this hidden option to persist the fact
        # that we have a non-adaptable layer with a
        # manually set vocab.
        self._has_input_vocabulary = kwargs.pop(
            "has_input_vocabulary", (vocabulary is not None)
        )
        vocabulary_size = kwargs.pop("vocabulary_size", None)

        super().__init__(name=name, **kwargs)

        self._lookup_layer = StringLookup(
            max_tokens=max_tokens,
            vocabulary=vocabulary,
            idf_weights=idf_weights,
            pad_to_max_tokens=pad_to_max_tokens,
            mask_token="",
            output_mode=output_mode,
            sparse=sparse,
            has_input_vocabulary=self._has_input_vocabulary,
            encoding=encoding,
            vocabulary_size=vocabulary_size,
        )
        self._convert_input_args = False
        self._allow_non_tensor_positional_args = True
        self.supports_jit = False

    @property
    def compute_dtype(self):
        return "string"

    @property
    def variable_dtype(self):
        return "string"

    def build(self, input_shape=None):
        pass

    def compute_output_shape(self, input_shape):
        if self._output_mode == "int":
            return (input_shape[0], self._output_sequence_length)
        if self._split is None:
            if len(input_shape) <= 1:
                input_shape = tuple(input_shape) + (1,)
        else:
            input_shape = tuple(input_shape) + (None,)
        return self._lookup_layer.compute_output_shape(input_shape)

    def compute_output_spec(self, inputs):
        output_shape = self.compute_output_shape(inputs.shape)
        if self._output_mode == "int":
            output_dtype = "int64"
        else:
            output_dtype = backend.floatx()
        return backend.KerasTensor(output_shape, dtype=output_dtype)

    def adapt(self, data, batch_size=None, steps=None):
        """Computes a vocabulary of string terms from tokens in a dataset.

        Calling `adapt()` on a `TextVectorization` layer is an alternative to
        passing in a precomputed vocabulary on construction via the `vocabulary`
        argument. A `TextVectorization` layer should always be either adapted
        over a dataset or supplied with a vocabulary.

        During `adapt()`, the layer will build a vocabulary of all string tokens
        seen in the dataset, sorted by occurrence count, with ties broken by
        sort order of the tokens (high to low). At the end of `adapt()`, if
        `max_tokens` is set, the vocabulary will be truncated to `max_tokens`
        size. For example, adapting a layer with `max_tokens=1000` will compute
        the 1000 most frequent tokens occurring in the input dataset. If
        `output_mode='tf-idf'`, `adapt()` will also learn the document
        frequencies of each token in the input dataset.

        Arguments:
            data: The data to train on. It can be passed either as a
                batched `tf.data.Dataset`, as a list of strings,
                or as a NumPy array.
            steps: Integer or `None`.
                Total number of steps (batches of samples) to process.
                If `data` is a `tf.data.Dataset`, and `steps` is `None`,
                `adapt()` will run until the input dataset is exhausted.
                When passing an infinitely
                repeating dataset, you must specify the `steps` argument. This
                argument is not supported with array inputs or list inputs.
        """
        self.reset_state()
        if isinstance(data, tf.data.Dataset):
            if steps is not None:
                data = data.take(steps)
            for batch in data:
                self.update_state(batch)
        else:
            data = tf_utils.ensure_tensor(data, dtype="string")
            if data.shape.rank == 1:
                # A plain list of strings
                # is treated as as many documents
                data = tf.expand_dims(data, -1)
            self.update_state(data)
        self.finalize_state()

    def update_state(self, data):
        self._lookup_layer.update_state(self._preprocess(data))

    def finalize_state(self):
        self._lookup_layer.finalize_state()

    def reset_state(self):
        self._lookup_layer.reset_state()

    def get_vocabulary(self, include_special_tokens=True):
        """Returns the current vocabulary of the layer.

        Args:
            include_special_tokens: If `True`, the returned vocabulary
                will include the padding and OOV tokens,
                and a term's index in the vocabulary will equal
                the term's index when calling the layer. If `False`, the
                returned vocabulary will not include any padding
                or OOV tokens.
        """
        return self._lookup_layer.get_vocabulary(include_special_tokens)

    def vocabulary_size(self):
        """Gets the current size of the layer's vocabulary.

        Returns:
            The integer size of the vocabulary, including optional
            mask and OOV indices.
        """
        return self._lookup_layer.vocabulary_size()

    def get_config(self):
        config = {
            "max_tokens": self._lookup_layer.max_tokens,
            "standardize": self._standardize,
            "split": self._split,
            "ngrams": self._ngrams_arg,
            "output_mode": self._output_mode,
            "output_sequence_length": self._output_sequence_length,
            "pad_to_max_tokens": self._lookup_layer.pad_to_max_tokens,
            "sparse": self._lookup_layer.sparse,
            "ragged": self._ragged,
            "vocabulary": listify_tensors(self._lookup_layer.input_vocabulary),
            "idf_weights": listify_tensors(
                self._lookup_layer.input_idf_weights
            ),
            "encoding": self._encoding,
            "vocabulary_size": self.vocabulary_size(),
        }
        base_config = super().get_config()
        return {**base_config, **config}

    @classmethod
    def from_config(cls, config):
        if not isinstance(config["standardize"], str):
            config["standardize"] = serialization_lib.deserialize_keras_object(
                config["standardize"]
            )
        if not isinstance(config["split"], str):
            config["split"] = serialization_lib.deserialize_keras_object(
                config["split"]
            )

        if isinstance(config["ngrams"], list):
            config["ngrams"] = tuple(config["ngrams"])

        return cls(**config)

    def set_vocabulary(self, vocabulary, idf_weights=None):
        """Sets vocabulary (and optionally document frequency) for this layer.

        This method sets the vocabulary and IDF weights for this layer directly,
        instead of analyzing a dataset through `adapt()`. It should be used
        whenever the vocab (and optionally document frequency) information is
        already known. If vocabulary data is already present in the layer, this
        method will replace it.

        Args:
            vocabulary: Either an array or a string path to a text file.
                If passing an array, can pass a tuple, list, 1D NumPy array,
                or 1D tensor containing the vocabulary terms.
                If passing a file path, the file should contain one line
                per term in the vocabulary.
            idf_weights: A tuple, list, 1D NumPy array, or 1D tensor of inverse
                document frequency weights with equal length to vocabulary.
                Must be set if `output_mode` is `"tf_idf"`.
                Should not be set otherwise.
        """
        self._lookup_layer.set_vocabulary(vocabulary, idf_weights=idf_weights)

    def _preprocess(self, inputs):
        inputs = tf_utils.ensure_tensor(inputs, dtype=tf.string)
        if self._standardize in ("lower", "lower_and_strip_punctuation"):
            inputs = tf.strings.lower(inputs)
        if self._standardize in (
            "strip_punctuation",
            "lower_and_strip_punctuation",
        ):
            inputs = tf.strings.regex_replace(
                inputs, r'[!"#$%&()\*\+,-\./:;<=>?@\[\\\]^_`{|}~\']', ""
            )
        if callable(self._standardize):
            inputs = self._standardize(inputs)

        if self._split is not None:
            # If we are splitting, we validate that the 1st axis is of dimension
            # 1 and so can be squeezed out. We do this here instead of after
            # splitting for performance reasons - it's more expensive to squeeze
            # a ragged tensor.
            if inputs.shape.rank > 1:
                if inputs.shape[-1] != 1:
                    raise ValueError(
                        "When using `TextVectorization` to tokenize strings, "
                        "the input rank must be 1 or the last shape dimension "
                        f"must be 1. Received: inputs.shape={inputs.shape} "
                        f"with rank={inputs.shape.rank}"
                    )
                else:
                    inputs = tf.squeeze(inputs, axis=-1)
            if self._split == "whitespace":
                # This treats multiple whitespaces as one whitespace, and strips
                # leading and trailing whitespace.
                inputs = tf.strings.split(inputs)
            elif self._split == "character":
                inputs = tf.strings.unicode_split(inputs, "UTF-8")
            elif callable(self._split):
                inputs = self._split(inputs)

        # Note that 'inputs' here can be either ragged or dense depending on the
        # configuration choices for this Layer. The strings.ngrams op, however,
        # does support both ragged and dense inputs.
        if self._ngrams is not None:
            inputs = tf.strings.ngrams(
                inputs, ngram_width=self._ngrams, separator=" "
            )
        return inputs

    def call(self, inputs):
        if not isinstance(
            inputs, (tf.Tensor, tf.RaggedTensor, np.ndarray, list, tuple)
        ):
            inputs = tf.convert_to_tensor(backend.convert_to_numpy(inputs))

        inputs = self._preprocess(inputs)

        # If we're not doing any output processing, return right away.
        if self._output_mode is None:
            outputs = inputs

        lookup_data = self._lookup_layer.call(inputs)

        # For non-int output, we can return directly from the underlying layer.
        if self._output_mode != "int":
            return backend_utils.convert_tf_tensor(lookup_data)

        # If we have a ragged tensor, we can pad during the conversion to dense.
        if isinstance(lookup_data, tf.RaggedTensor) and not self._ragged:
            shape = lookup_data.shape.as_list()
            # If output sequence length is None, to_tensor will pad the last
            # dimension to the bounding shape of the ragged dimension.
            shape[-1] = self._output_sequence_length
            outputs = lookup_data.to_tensor(default_value=0, shape=shape)
        # If we have a dense tensor, we need to pad/trim directly.
        elif self._output_sequence_length is not None:
            # Maybe trim the output.
            outputs = lookup_data[..., : self._output_sequence_length]

            # Maybe pad the output. We need to be careful to use dynamic shape
            # here as required_space_to_batch_paddings requires a fully known
            # shape.
            if not self._ragged:
                shape = tf.shape(outputs)
                padded_shape = tf.concat(
                    (shape[:-1], [self._output_sequence_length]), 0
                )
                padding, _ = tf.required_space_to_batch_paddings(
                    shape, padded_shape
                )
                outputs = tf.pad(outputs, padding)
                # Because `tf.pad` used a dynamic shape, the output shape is
                # dynamic. Apply the known static `_output_sequence_length`.
                static_padded_shape = lookup_data.shape.as_list()
                static_padded_shape[-1] = self._output_sequence_length
                outputs.set_shape(static_padded_shape)
        else:
            outputs = lookup_data

        return backend_utils.convert_tf_tensor(outputs)

    def save_own_variables(self, store):
        self._lookup_layer.save_own_variables(store)

    def load_own_variables(self, store):
        self._lookup_layer.load_own_variables(store)

    def save_assets(self, dir_path):
        self._lookup_layer.save_assets(dir_path)

    def load_assets(self, dir_path):
        self._lookup_layer.load_assets(dir_path)
