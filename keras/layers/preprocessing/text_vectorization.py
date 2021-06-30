# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Keras text vectorization preprocessing layer."""

import tensorflow.compat.v2 as tf
# pylint: disable=g-classes-have-attributes

import numpy as np
from keras import backend
from keras.engine import base_preprocessing_layer
from keras.layers.preprocessing import index_lookup
from keras.layers.preprocessing import string_lookup
from keras.utils import layer_utils
from keras.utils import tf_utils
from tensorflow.python.util.tf_export import keras_export

LOWER_AND_STRIP_PUNCTUATION = "lower_and_strip_punctuation"

SPLIT_ON_WHITESPACE = "whitespace"

TF_IDF = index_lookup.TF_IDF
INT = index_lookup.INT
MULTI_HOT = index_lookup.MULTI_HOT
COUNT = index_lookup.COUNT

# This is an explicit regex of all the tokens that will be stripped if
# LOWER_AND_STRIP_PUNCTUATION is set. If an application requires other
# stripping, a Callable should be passed into the 'standardize' arg.
DEFAULT_STRIP_REGEX = r'[!"#$%&()\*\+,-\./:;<=>?@\[\\\]^_`{|}~\']'

# The string tokens in the extracted vocabulary
_VOCAB_NAME = "vocab"
# The inverse-document-frequency weights
_IDF_NAME = "idf"
# The IDF data for the OOV token
_OOV_IDF_NAME = "oov_idf"

# The string tokens in the full vocabulary
_ACCUMULATOR_VOCAB_NAME = "vocab"
# The total counts of each token in the vocabulary
_ACCUMULATOR_COUNTS_NAME = "counts"
# The number of documents / examples that each token appears in.
_ACCUMULATOR_DOCUMENT_COUNTS = "document_counts"
# The total number of documents / examples in the dataset.
_ACCUMULATOR_NUM_DOCUMENTS = "num_documents"


@keras_export(
    "keras.layers.TextVectorization",
    "keras.layers.experimental.preprocessing.TextVectorization",
    v1=[])
class TextVectorization(base_preprocessing_layer.PreprocessingLayer):
  """Text vectorization layer.

  This layer has basic options for managing text in a Keras model. It
  transforms a batch of strings (one example = one string) into either a list of
  token indices (one example = 1D tensor of integer token indices) or a dense
  representation (one example = 1D tensor of float values representing data
  about the example's tokens).

  If desired, the user can call this layer's `adapt()` method on a dataset.
  When this layer is adapted, it will analyze the dataset, determine the
  frequency of individual string values, and create a 'vocabulary' from them.
  This vocabulary can have unlimited size or be capped, depending on the
  configuration options for this layer; if there are more unique values in the
  input than the maximum vocabulary size, the most frequent terms will be used
  to create the vocabulary.

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
     serializables (see `tf.keras.utils.register_keras_serializable` for more
     details).
  2. When using a custom callable for `standardize`, the data received
     by the callable will be exactly as passed to this layer. The callable
     should return a tensor of the same shape as the input.
  3. When using a custom callable for `split`, the data received by the
     callable will have the 1st dimension squeezed out - instead of
     `[["string to split"], ["another string to split"]]`, the Callable will
     see `["string to split", "another string to split"]`. The callable should
     return a Tensor with the first dimension containing the split tokens -
     in this example, we should see something like `[["string", "to",
     "split"], ["another", "string", "to", "split"]]`. This makes the callable
     site natively compatible with `tf.strings.split()`.

  Args:
    max_tokens: The maximum size of the vocabulary for this layer. If None,
      there is no cap on the size of the vocabulary. Note that this vocabulary
      contains 1 OOV token, so the effective number of tokens is `(max_tokens -
      1 - (1 if output_mode == "int" else 0))`.
    standardize: Optional specification for standardization to apply to the
      input text. Values can be None (no standardization),
      `"lower_and_strip_punctuation"` (lowercase and remove punctuation) or a
      Callable. Default is `"lower_and_strip_punctuation"`.
    split: Optional specification for splitting the input text. Values can be
      None (no splitting), `"whitespace"` (split on ASCII whitespace), or a
      Callable. The default is `"whitespace"`.
    ngrams: Optional specification for ngrams to create from the possibly-split
      input text. Values can be None, an integer or tuple of integers; passing
      an integer will create ngrams up to that integer, and passing a tuple of
      integers will create ngrams for the specified values in the tuple. Passing
      None means that no ngrams will be created.
    output_mode: Optional specification for the output of the layer. Values can
      be `"int"`, `"multi_hot"`, `"count"` or `"tf_idf"`, configuring the layer
      as follows:
        - `"int"`: Outputs integer indices, one integer index per split string
          token. When `output_mode == "int"`, 0 is reserved for masked
          locations; this reduces the vocab size to
          `max_tokens - 2` instead of `max_tokens - 1`.
        - `"multi_hot"`: Outputs a single int array per batch, of either
          vocab_size or max_tokens size, containing 1s in all elements where the
          token mapped to that index exists at least once in the batch item.
        - `"count"`: Like `"multi_hot"`, but the int array contains a count of
          the number of times the token at that index appeared in the
          batch item.
        - `"tf_idf"`: Like `"multi_hot"`, but the TF-IDF algorithm is applied to
          find the value in each token slot.
      For `"int"` output, any shape of input and output is supported. For all
      other output modes, currently only rank 1 inputs (and rank 2 outputs after
      splitting) are supported.
    output_sequence_length: Only valid in INT mode. If set, the output will have
      its time dimension padded or truncated to exactly `output_sequence_length`
      values, resulting in a tensor of shape
      `(batch_size, output_sequence_length)` regardless of how many tokens
      resulted from the splitting step. Defaults to None.
    pad_to_max_tokens: Only valid in  `"multi_hot"`, `"count"`, and `"tf_idf"`
      modes. If True, the output will have its feature axis padded to
      `max_tokens` even if the number of unique tokens in the vocabulary is less
      than max_tokens, resulting in a tensor of shape `(batch_size, max_tokens)`
      regardless of vocabulary size. Defaults to False.
    vocabulary: Optional. Either an array of strings or a string path to a text
      file. If passing an array, can pass a tuple, list, 1D numpy array, or 1D
      tensor containing the string vocbulary terms. If passing a file path, the
      file should contain one line per term in the vocabulary. If this argument
      is set, there is no need to `adapt` the layer.

  Example:

  This example instantiates a `TextVectorization` layer that lowercases text,
  splits on whitespace, strips punctuation, and outputs integer vocab indices.

  >>> text_dataset = tf.data.Dataset.from_tensor_slices(["foo", "bar", "baz"])
  >>> max_features = 5000  # Maximum vocab size.
  >>> max_len = 4  # Sequence length to pad the outputs to.
  >>>
  >>> # Create the layer.
  >>> vectorize_layer = tf.keras.layers.TextVectorization(
  ...  max_tokens=max_features,
  ...  output_mode='int',
  ...  output_sequence_length=max_len)
  >>>
  >>> # Now that the vocab layer has been created, call `adapt` on the text-only
  >>> # dataset to create the vocabulary. You don't have to batch, but for large
  >>> # datasets this means we're not keeping spare copies of the dataset.
  >>> vectorize_layer.adapt(text_dataset.batch(64))
  >>>
  >>> # Create the model that uses the vectorize text layer
  >>> model = tf.keras.models.Sequential()
  >>>
  >>> # Start by creating an explicit input layer. It needs to have a shape of
  >>> # (1,) (because we need to guarantee that there is exactly one string
  >>> # input per batch), and the dtype needs to be 'string'.
  >>> model.add(tf.keras.Input(shape=(1,), dtype=tf.string))
  >>>
  >>> # The first layer in our model is the vectorization layer. After this
  >>> # layer, we have a tensor of shape (batch_size, max_len) containing vocab
  >>> # indices.
  >>> model.add(vectorize_layer)
  >>>
  >>> # Now, the model can map strings to integers, and you can add an embedding
  >>> # layer to map these integers to learned embeddings.
  >>> input_data = [["foo qux bar"], ["qux baz"]]
  >>> model.predict(input_data)
  array([[2, 1, 4, 0],
         [1, 3, 0, 0]])

  Example:

  This example instantiates a `TextVectorization` layer by passing a list
  of vocabulary terms to the layer's `__init__()` method.

  >>> vocab_data = ["earth", "wind", "and", "fire"]
  >>> max_len = 4  # Sequence length to pad the outputs to.
  >>>
  >>> # Create the layer, passing the vocab directly. You can also pass the
  >>> # vocabulary arg a path to a file containing one vocabulary word per
  >>> # line.
  >>> vectorize_layer = tf.keras.layers.TextVectorization(
  ...  max_tokens=max_features,
  ...  output_mode='int',
  ...  output_sequence_length=max_len,
  ...  vocabulary=vocab_data)
  >>>
  >>> # Because we've passed the vocabulary directly, we don't need to adapt
  >>> # the layer - the vocabulary is already set. The vocabulary contains the
  >>> # padding token ('') and OOV token ('[UNK]') as well as the passed tokens.
  >>> vectorize_layer.get_vocabulary()
  ['', '[UNK]', 'earth', 'wind', 'and', 'fire']

  """
  # TODO(momernick): Add an examples section to the docstring.

  def __init__(self,
               max_tokens=None,
               standardize="lower_and_strip_punctuation",
               split="whitespace",
               ngrams=None,
               output_mode="int",
               output_sequence_length=None,
               pad_to_max_tokens=False,
               vocabulary=None,
               **kwargs):

    # This layer only applies to string processing, and so should only have
    # a dtype of 'string'.
    if "dtype" in kwargs and kwargs["dtype"] != tf.string:
      raise ValueError("TextVectorization may only have a dtype of string.")
    elif "dtype" not in kwargs:
      kwargs["dtype"] = tf.string

    # 'standardize' must be one of (None, LOWER_AND_STRIP_PUNCTUATION, callable)
    layer_utils.validate_string_arg(
        standardize,
        allowable_strings=(LOWER_AND_STRIP_PUNCTUATION),
        layer_name="TextVectorization",
        arg_name="standardize",
        allow_none=True,
        allow_callables=True)

    # 'split' must be one of (None, SPLIT_ON_WHITESPACE, callable)
    layer_utils.validate_string_arg(
        split,
        allowable_strings=(SPLIT_ON_WHITESPACE),
        layer_name="TextVectorization",
        arg_name="split",
        allow_none=True,
        allow_callables=True)

    # Support deprecated names for output_modes.
    if output_mode == "binary":
      output_mode = MULTI_HOT
    if output_mode == "tf-idf":
      output_mode = TF_IDF
    # 'output_mode' must be one of (None, INT, COUNT, MULTI_HOT, TF_IDF)
    layer_utils.validate_string_arg(
        output_mode,
        allowable_strings=(INT, COUNT, MULTI_HOT, TF_IDF),
        layer_name="TextVectorization",
        arg_name="output_mode",
        allow_none=True)

    # 'ngrams' must be one of (None, int, tuple(int))
    if not (ngrams is None or
            isinstance(ngrams, int) or
            isinstance(ngrams, tuple) and
            all(isinstance(item, int) for item in ngrams)):
      raise ValueError(("`ngrams` must be None, an integer, or a tuple of "
                        "integers. Got %s") % (ngrams,))

    # 'output_sequence_length' must be one of (None, int) and is only
    # set if output_mode is INT.
    if (output_mode == INT and not (isinstance(output_sequence_length, int) or
                                    (output_sequence_length is None))):
      raise ValueError("`output_sequence_length` must be either None or an "
                       "integer when `output_mode` is 'int'. "
                       "Got %s" % output_sequence_length)

    if output_mode != INT and output_sequence_length is not None:
      raise ValueError("`output_sequence_length` must not be set if "
                       "`output_mode` is not 'int'.")

    self._max_tokens = max_tokens
    self._standardize = standardize
    self._split = split
    self._ngrams_arg = ngrams
    if isinstance(ngrams, int):
      self._ngrams = tuple(range(1, ngrams + 1))
    else:
      self._ngrams = ngrams

    self._output_mode = output_mode
    self._output_sequence_length = output_sequence_length
    # Drop deprecated config options.
    kwargs.pop("vocabulary_size", None)

    super().__init__(**kwargs)
    base_preprocessing_layer.keras_kpl_gauge.get_cell("TextVectorization").set(
        True)

    self._index_lookup_layer = string_lookup.StringLookup(
        max_tokens=max_tokens,
        vocabulary=vocabulary,
        pad_to_max_tokens=pad_to_max_tokens,
        mask_token="",
        output_mode=output_mode if output_mode is not None else INT)

  def compute_output_shape(self, input_shape):
    if self._output_mode == INT:
      return tf.TensorShape([input_shape[0], self._output_sequence_length])

    if self._split is None:
      if len(input_shape) <= 1:
        input_shape = tuple(input_shape) + (1,)
    else:
      input_shape = tuple(input_shape) + (None,)
    return self._index_lookup_layer.compute_output_shape(input_shape)

  def compute_output_signature(self, input_spec):
    output_shape = self.compute_output_shape(input_spec.shape.as_list())
    output_dtype = (tf.int64 if self._output_mode == INT
                    else backend.floatx())
    return tf.TensorSpec(shape=output_shape, dtype=output_dtype)

  def update_state(self, data):
    self._index_lookup_layer.update_state(self._preprocess(data))

  def finalize_state(self):
    self._index_lookup_layer.finalize_state()

  def reset_state(self):  # pylint: disable=method-hidden
    self._index_lookup_layer.reset_state()

  def get_vocabulary(self, include_special_tokens=True):
    """Returns the current vocabulary of the layer.

    Args:
      include_special_tokens: If True, the returned vocabulary will include
        the padding and OOV tokens, and a term's index in the vocabulary will
        equal the term's index when calling the layer. If False, the returned
        vocabulary will not include any padding or OOV tokens.
    """
    return self._index_lookup_layer.get_vocabulary(include_special_tokens)

  def vocabulary_size(self):
    """Gets the current size of the layer's vocabulary.

    Returns:
      The integer size of the voculary, including optional mask and oov indices.
    """
    return self._index_lookup_layer.vocabulary_size()

  def get_config(self):
    # This does not include the 'vocabulary' arg, since if the vocab was passed
    # at init time it's now stored in variable state - we don't need to
    # pull it off disk again.
    config = {
        "max_tokens": self._index_lookup_layer.max_tokens,
        "standardize": self._standardize,
        "split": self._split,
        "ngrams": self._ngrams_arg,
        "output_mode": self._output_mode,
        "output_sequence_length": self._output_sequence_length,
        "pad_to_max_tokens": self._index_lookup_layer.pad_to_max_tokens,
    }
    base_config = super(TextVectorization, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

  def set_vocabulary(self, vocabulary, idf_weights=None):
    """Sets vocabulary (and optionally document frequency) data for this layer.

    This method sets the vocabulary and idf weights for this layer directly,
    instead of analyzing a dataset through 'adapt'. It should be used whenever
    the vocab (and optionally document frequency) information is already known.
    If vocabulary data is already present in the layer, this method will replace
    it.

    Args:
      vocabulary: Either an array or a string path to a text file. If passing an
        array, can pass a tuple, list, 1D numpy array, or 1D tensor containing
        the vocbulary terms. If passing a file path, the file should contain one
        line per term in the vocabulary.
      idf_weights: A tuple, list, 1D numpy array, or 1D tensor of inverse
        document frequency weights with equal length to vocabulary. Must be set
        if `output_mode` is `"tf_idf"`. Should not be set otherwise.

    Raises:
      ValueError: If there are too many inputs, the inputs do not match, or
        input data is missing.
      RuntimeError: If the vocabulary cannot be set when this function is
        called. This happens when `"multi_hot"`, `"count"`, and "tf_idf" modes,
        if `pad_to_max_tokens` is False and the layer itself has already been
        called.
    """
    self._index_lookup_layer.set_vocabulary(vocabulary, idf_weights=idf_weights)

  def build(self, input_shape):
    # We have to use 'and not ==' here, because input_shape[1] !/== 1 can result
    # in None for undefined shape axes. If using 'and !=', this causes the
    # expression to evaluate to False instead of True if the shape is undefined;
    # the expression needs to evaluate to True in that case.
    if self._split is not None:
      if input_shape.ndims > 1 and not input_shape[-1] == 1:  # pylint: disable=g-comparison-negation
        raise RuntimeError(
            "When using TextVectorization to tokenize strings, the innermost "
            "dimension of the input array must be 1, got shape "
            "{}".format(input_shape))

    super(TextVectorization, self).build(input_shape)

  def _preprocess(self, inputs):
    if self._standardize == LOWER_AND_STRIP_PUNCTUATION:
      if tf_utils.is_ragged(inputs):
        lowercase_inputs = tf.ragged.map_flat_values(
            tf.strings.lower, inputs)
        # Depending on configuration, we may never touch the non-data tensor
        # in the ragged inputs tensor. If that is the case, and this is the
        # only layer in the keras model, running it will throw an error.
        # To get around this, we wrap the result in an identity.
        lowercase_inputs = tf.identity(lowercase_inputs)
      else:
        lowercase_inputs = tf.strings.lower(inputs)
      inputs = tf.strings.regex_replace(lowercase_inputs, DEFAULT_STRIP_REGEX,
                                        "")
    elif callable(self._standardize):
      inputs = self._standardize(inputs)
    elif self._standardize is not None:
      raise ValueError(("%s is not a supported standardization. "
                        "TextVectorization supports the following options "
                        "for `standardize`: None, "
                        "'lower_and_strip_punctuation', or a "
                        "Callable.") % self._standardize)

    if self._split is not None:
      # If we are splitting, we validate that the 1st axis is of dimension 1 and
      # so can be squeezed out. We do this here instead of after splitting for
      # performance reasons - it's more expensive to squeeze a ragged tensor.
      if inputs.shape.ndims > 1:
        inputs = tf.squeeze(inputs, axis=-1)
      if self._split == SPLIT_ON_WHITESPACE:
        # This treats multiple whitespaces as one whitespace, and strips leading
        # and trailing whitespace.
        inputs = tf.strings.split(inputs)
      elif callable(self._split):
        inputs = self._split(inputs)
      else:
        raise ValueError(
            ("%s is not a supported splitting."
             "TextVectorization supports the following options "
             "for `split`: None, 'whitespace', or a Callable.") % self._split)

    # Note that 'inputs' here can be either ragged or dense depending on the
    # configuration choices for this Layer. The strings.ngrams op, however, does
    # support both ragged and dense inputs.
    if self._ngrams is not None:
      inputs = tf.strings.ngrams(
          inputs, ngram_width=self._ngrams, separator=" ")

    return inputs

  def call(self, inputs):
    if isinstance(inputs, (list, tuple, np.ndarray)):
      inputs = tf.convert_to_tensor(inputs)

    inputs = self._preprocess(inputs)

    # If we're not doing any output processing, return right away.
    if self._output_mode is None:
      return inputs

    lookup_data = self._index_lookup_layer(inputs)

    # For any non-int output, we can return directly from the underlying layer.
    if self._output_mode is not INT:
      return lookup_data

    # If we have a ragged tensor, we can pad during the conversion to dense.
    if tf_utils.is_ragged(lookup_data):
      shape = lookup_data.shape.as_list()
      # If output sequence length is None, to_tensor will pad the last dimension
      # to the bounding shape of the ragged dimension.
      shape[-1] = self._output_sequence_length
      return lookup_data.to_tensor(default_value=0, shape=shape)

    # If we have a dense tensor, we need to pad/trim directly.
    if self._output_sequence_length is not None:
      # Maybe trim the output.
      lookup_data = lookup_data[..., :self._output_sequence_length]

      # Maybe pad the output. We need to be careful to use dynamic shape here as
      # required_space_to_batch_paddings requires a fully known shape.
      shape = tf.shape(lookup_data)
      padded_shape = tf.concat((shape[:-1], [self._output_sequence_length]), 0)
      padding, _ = tf.required_space_to_batch_paddings(shape, padded_shape)
      return tf.pad(lookup_data, padding)

    return lookup_data
