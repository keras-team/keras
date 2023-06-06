import numpy as np
import tensorflow as tf

from keras_core import backend
from keras_core.api_export import keras_core_export
from keras_core.layers.layer import Layer
from keras_core.saving import serialization_lib
from keras_core.utils import backend_utils


@keras_core_export("keras_core.layers.TextVectorization")
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
       serializables (see `keras_core.saving.register_keras_serializable`
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

    **Note:** This layer wraps `tf.keras.layers.TextVectorization`. It cannot
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
            resulted from the splitting step. Defaults to `None`.
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
    >>> vectorize_layer = keras_core.layers.TextVectorization(
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
        super().__init__(name=name)
        if sparse and backend.backend() != "tensorflow":
            raise ValueError(
                "`sparse` can only be set to True with the "
                "TensorFlow backend."
            )
        if ragged and backend.backend() != "tensorflow":
            raise ValueError(
                "`ragged` can only be set to True with the "
                "TensorFlow backend."
            )
        self.layer = tf.keras.layers.TextVectorization(
            max_tokens=max_tokens,
            standardize=standardize,
            split=split,
            ngrams=ngrams,
            output_mode=output_mode,
            output_sequence_length=output_sequence_length,
            pad_to_max_tokens=pad_to_max_tokens,
            vocabulary=vocabulary,
            idf_weights=idf_weights,
            sparse=sparse,
            ragged=ragged,
            encoding=encoding,
            name=name,
            **kwargs,
        )
        self._convert_input_args = False
        self._allow_non_tensor_positional_args = True
        self.supports_jit = False

    def compute_output_shape(self, input_shape):
        return tuple(self.layer.compute_output_shape(input_shape))

    def adapt(self, data, batch_size=None, steps=None):
        """Computes a vocabulary of string terms from tokens in a dataset.

        Calling `adapt()` on a `TextVectorization` layer is an alternative to
        passing in a precomputed vocabulary on construction via the `vocabulary`
        argument. A `TextVectorization` layer should always be either adapted
        over a dataset or supplied with a vocabulary.

        During `adapt()`, the layer will build a vocabulary of all string tokens
        seen in the dataset, sorted by occurrence count, with ties broken by
        sort order of the tokens (high to low). At the end of `adapt()`, if
        `max_tokens` is set, the vocabulary wil be truncated to `max_tokens`
        size. For example, adapting a layer with `max_tokens=1000` will compute
        the 1000 most frequent tokens occurring in the input dataset. If
        `output_mode='tf-idf'`, `adapt()` will also learn the document
        frequencies of each token in the input dataset.

        Arguments:
            data: The data to train on. It can be passed either as a
                batched `tf.data.Dataset`, as a list of strings,
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
        return self.layer.get_vocabulary(
            include_special_tokens=include_special_tokens
        )

    def vocabulary_size(self):
        """Gets the current size of the layer's vocabulary.

        Returns:
            The integer size of the vocabulary, including optional
            mask and OOV indices.
        """
        return self.layer.vocabulary_size()

    def get_config(self):
        return self.layer.get_config()

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
                or 1D tensor containing the vocbulary terms.
                If passing a file path, the file should contain one line
                per term in the vocabulary.
            idf_weights: A tuple, list, 1D NumPy array, or 1D tensor of inverse
                document frequency weights with equal length to vocabulary.
                Must be set if `output_mode` is `"tf_idf"`.
                Should not be set otherwise.
        """
        self.layer.set_vocabulary(vocabulary, idf_weights=idf_weights)

    def call(self, inputs):
        if not isinstance(inputs, (tf.Tensor, np.ndarray, list, tuple)):
            inputs = tf.convert_to_tensor(np.array(inputs))
        outputs = self.layer.call(inputs)
        if (
            backend.backend() != "tensorflow"
            and not backend_utils.in_tf_graph()
        ):
            outputs = backend.convert_to_tensor(outputs)
        return outputs

    def save_own_variables(self, store):
        self.layer.save_own_variables(store)

    def load_own_variables(self, store):
        self.layer.load_own_variables(store)

    def save_assets(self, dir_path):
        self.layer.save_assets(dir_path)

    def load_assets(self, dir_path):
        self.layer.load_assets(dir_path)
