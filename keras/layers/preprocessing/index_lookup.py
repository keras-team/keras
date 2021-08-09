# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""Keras index lookup preprocessing layer."""

# pylint: disable=g-classes-have-attributes

import collections

from keras import backend
from keras.engine import base_layer_utils
from keras.engine import base_preprocessing_layer
from keras.layers.preprocessing import category_encoding
from keras.saving.saved_model import layer_serialization
from keras.utils import layer_utils
from keras.utils import tf_utils
import numpy as np
import tensorflow.compat.v2 as tf
from tensorflow.python.platform import tf_logging as logging

INT = "int"
MULTI_HOT = "multi_hot"
ONE_HOT = "one_hot"
COUNT = "count"
TF_IDF = "tf_idf"

_VOCAB_NAME = "vocab"
_IDF_WEIGHTS_NAME = "idf_weights"


class NullInitializer(tf.lookup.KeyValueTensorInitializer):
  """A placeholder initializer for restoring this layer from a SavedModel."""

  def __init__(self, key_dtype, value_dtype):
    """Construct a table initializer object.

    Args:
      key_dtype: Type of the table keys.
      value_dtype: Type of the table values.
    """
    self._key_dtype = key_dtype
    self._value_dtype = value_dtype

  @property
  def key_dtype(self):
    """The expected table key dtype."""
    return self._key_dtype

  @property
  def value_dtype(self):
    """The expected table value dtype."""
    return self._value_dtype

  def initialize(self, table):
    """Returns the table initialization op."""
    pass


class VocabWeightHandler(base_layer_utils.TrackableWeightHandler):
  """Adds the vocabulary as a layer weight during serialization."""

  def __init__(self, lookup_layer):
    self._layer = lookup_layer
    self._dtype = lookup_layer.dtype
    self._distribute_strategy = tf.distribute.get_strategy()

  @property
  def num_tensors(self):
    return 1

  def set_weights(self, weights):
    tokens = tf.convert_to_tensor(weights[0], self._dtype)
    self._layer.lookup_table = self._layer._lookup_table_from_tokens(tokens)  # pylint: disable=protected-access

  def get_tensors(self):
    # Just save the non-config part of the vocab (no special tokens).
    tokens = self._layer.get_vocabulary(include_special_tokens=False)
    tokens = tf.convert_to_tensor(tokens, self._dtype)
    return [tokens]


class IndexLookup(base_preprocessing_layer.PreprocessingLayer):
  """Maps values from a vocabulary to integer indices.

  This layer translates a set of arbitrary hashables into an integer output via
  a table-based lookup, with optional out-of-vocabulary handling. This is the
  basis layer for both IntegerLookup and StringLookup; it holds the common
  logic but is not intended to be exported as part of the Keras API.

  Args:
    max_tokens: The maximum size of the vocabulary for this layer. If None,
      there is no cap on the size of the vocabulary. Note that this size
      includes the OOV and mask tokens.
    num_oov_indices: The number of out-of-vocabulary tokens to use. If this
      value is more than 1, OOV inputs are hashed to determine their OOV value.
      If this value is 0, OOV inputs will cause an error when calling the layer.
    mask_token: A token that represents masked inputs. When `output_mode` is
      `"int"`, the token is included in vocabulary and mapped to index 0. In
      other output modes, the token will not appear in the vocabulary and
      instances of the mask token in the input will be dropped. If set to None,
      no mask term will be added.
    oov_token: Only used when `invert` is True. The token to return for OOV
      indices.
    vocabulary: Optional. Either an array or a string path to a text file. If
      passing an array, can pass a tuple, list, 1D numpy array, or 1D tensor
      containing the vocbulary terms. If passing a file path, the file should
      contain one line per term in the vocabulary. If this argument is set,
      there is no need to `adapt` the layer.
    invert: Only valid when `output_mode` is `"int"`. If True, this layer will
      map indices to vocabulary items instead of mapping vocabulary items to
      indices. Default to False.
    output_mode: Specification for the output of the layer. Defaults to `"int"`.
      Values can be `"int"`, `"one_hot"`, `"multi_hot"`, `"count"`, or
      `"tf_idf"` configuring the layer as follows:
        - `"int"`: Return the raw integer indices of the input tokens.
        - `"one_hot"`: Encodes each individual element in the input into an
          array the same size as the vocabulary, containing a 1 at the element
          index. If the last dimension is size 1, will encode on that dimension.
          If the last dimension is not size 1, will append a new dimension for
          the encoded output.
        - `"multi_hot"`: Encodes each sample in the input into a single array
          the same size as the vocabulary, containing a 1 for each vocabulary
          term present in the sample. Treats the last dimension as the sample
          dimension, if input shape is (..., sample_length), output shape will
          be (..., num_tokens).
        - `"count"`: As `"multi_hot"`, but the int array contains a count of the
          number of times the token at that index appeared in the sample.
        - `"tf_idf"`: As `"multi_hot"`, but the TF-IDF algorithm is applied to
          find the value in each token slot.
    pad_to_max_tokens: Only valid when `output_mode` is `"multi_hot"`,
      `"count"`, or `"tf_idf"`. If True, the output will have its feature axis
      padded to `max_tokens` even if the number of unique tokens in the
      vocabulary is less than max_tokens, resulting in a tensor of shape
      [batch_size, max_tokens] regardless of vocabulary size. Defaults to False.
    sparse: Boolean. Only applicable to `"multi_hot"` and `"count"` output
      modes. If True, returns a `SparseTensor` instead of a dense `Tensor`.
      Defaults to False.
  """

  def __init__(self,
               max_tokens,
               num_oov_indices,
               mask_token,
               oov_token,
               vocabulary=None,
               invert=False,
               output_mode="int",
               sparse=False,
               pad_to_max_tokens=False,
               **kwargs):
    # If max_tokens is set, the value must be greater than 1 - otherwise we
    # are creating a 0-element vocab, which doesn't make sense.
    if max_tokens is not None and max_tokens <= 1:
      raise ValueError("If set, `max_tokens` must be greater than 1. "
                       "You passed `max_tokens={}`".format(max_tokens))

    if pad_to_max_tokens and max_tokens is None:
      raise ValueError("If pad_to_max_tokens is True, must set `max_tokens`. "
                       "You passed `max_tokens={}`".format(max_tokens))

    if num_oov_indices < 0:
      raise ValueError("`num_oov_indices` must be greater than or equal to 0. "
                       "You passed {}".format(num_oov_indices))

    # Support deprecated names for output_modes.
    if output_mode == "binary":
      output_mode = MULTI_HOT
    if output_mode == "tf-idf":
      output_mode = TF_IDF
    # 'output_mode' must be one of (INT, ONE_HOT, MULTI_HOT, COUNT, TF_IDF)
    layer_utils.validate_string_arg(
        output_mode,
        allowable_strings=(INT, ONE_HOT, MULTI_HOT, COUNT, TF_IDF),
        layer_name=self.__class__.__name__,
        arg_name="output_mode")

    if invert and output_mode != INT:
      raise ValueError("`output_mode` must be {} when `invert` is true. You "
                       "passed {}".format(INT, output_mode))

    self.invert = invert
    self.max_tokens = max_tokens
    self.num_oov_indices = num_oov_indices
    self.mask_token = mask_token
    self.oov_token = oov_token
    self.output_mode = output_mode
    self.sparse = sparse
    self.pad_to_max_tokens = pad_to_max_tokens
    self.input_vocabulary = None
    # IndexLookupLayerSavedModelSaver will clear the config config vocabulary to
    # restore the lookup table ops directly. We persist this hidden option to
    # persist the fact that we have have a non-adaptable layer with a manually
    # set vocabulary.
    self._has_input_vocabulary = kwargs.pop("has_input_vocabulary", False)
    self._frozen_vocab_size = None

    # Drop deprecated config options.
    kwargs.pop("vocabulary_size", None)
    kwargs.pop("has_static_table", None)

    super().__init__(**kwargs)

    if invert:
      self._key_dtype = tf.int64
      self._value_dtype = tf.as_dtype(self.dtype)
      mask_key = 0
      mask_value = mask_token
      self._default_value = self.oov_token
    else:
      self._key_dtype = tf.as_dtype(self.dtype)
      self._value_dtype = tf.int64
      mask_key = mask_token
      # Masks should map to 0 for int output and be dropped otherwise. Max ints
      # will be dropped from the bincount op.
      mask_value = 0 if self.output_mode == INT else tf.int64.max
      if self.num_oov_indices == 0:
        # If there are no OOV indices, we map OOV tokens to -1 and error out
        # during call if we find a negative index.
        self._default_value = -1
      elif self.num_oov_indices == 1:
        # If there is only one OOV index, we can set that index as the default
        # value of the index_lookup table.
        self._default_value = self._oov_start_index()
      else:
        # If we hav multiple OOV values, we need to do a further hashing step;
        # to make this easier, we set the OOV value to -1. (This lets us do a
        # vectorized add and cast to boolean to determine locations where we
        # need to do extra hashing.)
        self._default_value = -1
    if self.mask_token is not None:
      self._mask_key = tf.convert_to_tensor(mask_key, self._key_dtype)
      self._mask_value = tf.convert_to_tensor(mask_value, self._value_dtype)

    if self.output_mode == TF_IDF:
      self.idf_weights = tf.Variable(
          [0] * self._token_start_index(),
          shape=(None,),
          dtype=backend.floatx(),
          trainable=False)
      self.idf_weights_const = self.idf_weights.value()

    if vocabulary is not None:
      self.set_vocabulary(vocabulary)
    else:
      # When restoring from a keras SavedModel, the loading code will expect to
      # find and restore a lookup_table attribute on the layer. This table needs
      # to be uninitialized as a StaticHashTable cannot be initialized twice.
      self.lookup_table = self._uninitialized_lookup_table()
      if not self._has_input_vocabulary:
        # Add a custom weight handler to return the layers vocab as it's weight.
        self._add_trackable(VocabWeightHandler(self), False)
        # Set adapt state.
        self.token_counts = tf.lookup.experimental.MutableHashTable(
            key_dtype=self.dtype, value_dtype=tf.int64, default_value=0)
        if self.output_mode == TF_IDF:
          self.token_document_counts = tf.lookup.experimental.MutableHashTable(
              key_dtype=self.dtype, value_dtype=tf.int64, default_value=0)
          self.num_documents = tf.Variable(0, dtype=tf.int64, trainable=False)

  def compute_output_shape(self, input_shape):
    if self.output_mode == INT:
      return input_shape
    if self.pad_to_max_tokens:
      out_depth = self.max_tokens
    else:
      out_depth = self.vocabulary_size()
    return tf.TensorShape([input_shape[0], out_depth])

  def compute_output_signature(self, input_spec):
    output_shape = self.compute_output_shape(input_spec.shape.as_list())
    output_dtype = (
        self._value_dtype if self.output_mode == INT else backend.floatx())
    return tf.TensorSpec(shape=output_shape, dtype=output_dtype)

  def get_vocabulary(self, include_special_tokens=True):
    """Returns the current vocabulary of the layer.

    Args:
      include_special_tokens: If True, the returned vocabulary will include mask
        and OOV tokens, and a term's index in the vocabulary will equal the
        term's index when calling the layer. If False, the returned vocabulary
        will not include any mask or OOV tokens.
    """
    # The lookup table data will not be sorted, so we will create a inverted
    # lookup here, and use that to lookup a range of indices [0, vocab_size).
    if self.lookup_table.size() == 0:
      vocab, indices = [], []
    else:
      keys, values = self.lookup_table.export()
      vocab, indices = (values, keys) if self.invert else (keys, values)
      vocab, indices = (self._tensor_vocab_to_numpy(vocab), indices.numpy())
    lookup = collections.defaultdict(lambda: self.oov_token,
                                     zip(indices, vocab))
    vocab = [lookup[x] for x in range(self.vocabulary_size())]
    if self.mask_token is not None and self.output_mode == INT:
      vocab[0] = self.mask_token
    if not include_special_tokens:
      vocab = vocab[self._token_start_index():]
    return vocab

  def vocabulary_size(self):
    """Gets the current size of the layer's vocabulary.

    Returns:
      The integer size of the voculary, including optional mask and oov indices.
    """
    return int(self.lookup_table.size().numpy()) + self._token_start_index()

  def vocab_size(self):
    logging.warning("vocab_size is deprecated, please use vocabulary_size.")
    return self.vocabulary_size()

  def get_config(self):
    config = {
        "invert": self.invert,
        "max_tokens": self.max_tokens,
        "num_oov_indices": self.num_oov_indices,
        "oov_token": self.oov_token,
        "mask_token": self.mask_token,
        "output_mode": self.output_mode,
        "pad_to_max_tokens": self.pad_to_max_tokens,
        "vocabulary": self._make_serializable(self.input_vocabulary),
    }

    base_config = super().get_config()
    return dict(list(base_config.items()) + list(config.items()))

  def set_vocabulary(self, vocabulary, idf_weights=None):
    """Sets vocabulary (and optionally document frequency) data for this layer.

    This method sets the vocabulary and idf weights for this layer directly,
    instead of analyzing a dataset through `adapt`. It should be used whenever
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
        called. This happens when `"multi_hot"`, `"count"`, and `"tf_idf"`
        modes, if `pad_to_max_tokens` is False and the layer itself has already
        been called.
      RuntimeError: If a tensor vocabulary is passed outside of eager execution.
    """
    self.input_vocabulary = vocabulary
    self._has_input_vocabulary = True

    if self.output_mode != TF_IDF and idf_weights is not None:
      raise ValueError("`idf_weights` should only be set if output_mode is "
                       "TF_IDF. output_mode is {}.".format(self.output_mode))

    if isinstance(vocabulary, str):
      if not tf.io.gfile.exists(vocabulary):
        raise ValueError(
            "Vocabulary file {} does not exist.".format(vocabulary))
      if self.output_mode == TF_IDF:
        raise ValueError("output_mode `'tf_idf'` does not support loading a "
                         "vocabulary from file.")
      self.lookup_table = self._lookup_table_from_file(vocabulary)
      return

    if not tf.executing_eagerly() and (tf.is_tensor(vocabulary) or
                                       tf.is_tensor(idf_weights)):
      raise RuntimeError(
          "Cannot set a tensor vocabulary on {} layer {} when not executing "
          "eagerly. Create this layer or call `set_vocabulary` outside of "
          "any `tf.function`s and with eager execution enabled.".format(
              self.__class__.__name__, self.name))

    # TODO(mattdangerw): for better performance we should rewrite this entire
    # function to operate on tensors and convert vocabulary to a tensor here.
    if tf.is_tensor(vocabulary):
      vocabulary = self._tensor_vocab_to_numpy(vocabulary)
    elif isinstance(vocabulary, (list, tuple)):
      vocabulary = np.array(vocabulary)
    if tf.is_tensor(idf_weights):
      idf_weights = idf_weights.numpy()
    elif isinstance(idf_weights, (list, tuple)):
      idf_weights = np.array(idf_weights)

    if vocabulary.size == 0:
      raise ValueError(
          "Cannot set an empty vocabulary, you passed {}.".format(vocabulary))

    oov_start = self._oov_start_index()
    token_start = self._token_start_index()
    should_have_mask = (oov_start > 0)
    has_mask = should_have_mask and vocabulary[0] == self.mask_token

    should_have_oov = (self.num_oov_indices > 0)
    expected_oov = [self.oov_token] * self.num_oov_indices
    found_oov = vocabulary[oov_start:token_start]
    has_oov = should_have_oov and np.array_equal(found_oov, expected_oov)

    if all([should_have_mask, has_mask, should_have_oov]) and not has_oov:
      raise ValueError(
          "Invalid vocabulary format. The layer was created with "
          "`mask_token={mask}` and `oov_token={oov}`. These tokens should be "
          "included in the provided vocabulary. The passed vocabulary has the "
          "correct mask token `{mask}` at index 0, but does not have the OOV "
          "token `{oov}` in indices [{start}:{end}]. Instead, we found "
          "`{found}`. Was this vocabulary generated by a layer with "
          "incompatible settings?".format(
              mask=self.mask_token,
              oov=self.oov_token,
              start=oov_start,
              end=token_start,
              found=found_oov))

    if all([should_have_oov, has_oov, should_have_mask]) and not has_mask:
      raise ValueError(
          "Invalid vocabulary format. The layer was created with "
          "`mask_token={mask}` and `oov_token={oov}`. These tokens should be "
          "included in the provided vocabulary. The passed vocabulary has the "
          "correct OOV token `{oov}` at indices [{start}:{end}], but does not "
          "have the mask token `{mask}` in index 0. Instead, we found "
          "`{found}`. Was this vocabulary generated by a layer with "
          "incompatible settings?".format(
              mask=self.mask_token,
              oov=self.oov_token,
              start=oov_start,
              end=token_start,
              found=vocabulary[0]))

    found_special_tokens = has_oov or has_mask
    if found_special_tokens:
      tokens = vocabulary[token_start:]
    else:
      tokens = vocabulary

    repeated_tokens = self._find_repeated_tokens(tokens)
    if repeated_tokens:
      raise ValueError("The passed vocabulary has at least one repeated "
                       "term. Please uniquify your dataset. The repeated terms "
                       "are {}".format(repeated_tokens))

    if self.mask_token in tokens:
      mask_index = np.argwhere(tokens == self.mask_token)[0]
      raise ValueError("Reserved mask token {} was found in the passed "
                       "vocabulary at index {}. Please either remove the "
                       "reserved token from the vocabulary or change the "
                       "mask token for this layer.".format(
                           self.mask_token, mask_index))
    if self.oov_token in tokens:
      oov_index = np.argwhere(tokens == self.oov_token)[0]
      raise ValueError("Reserved OOV token {} was found in the passed "
                       "vocabulary at index {}. Please either remove the "
                       "reserved token from the vocabulary or change the "
                       "OOV token for this layer.".format(
                           self.oov_token, oov_index))

    new_vocab_size = token_start + len(tokens)
    if self.max_tokens is not None and (new_vocab_size > self.max_tokens):
      raise ValueError(
          "Attempted to set a vocabulary larger than the maximum vocab size. "
          "Passed vocab size is {}, max vocab size is {}.".format(
              new_vocab_size, self.max_tokens))
    self.lookup_table = self._lookup_table_from_tokens(tokens)

    if self.output_mode == TF_IDF:
      if idf_weights is None:
        raise ValueError("`idf_weights` must be set if output_mode is TF_IDF")
      if len(vocabulary) != len(idf_weights):
        raise ValueError("`idf_weights` must be the same length as vocabulary. "
                         "len(idf_weights) is {}, len(vocabulary) is {}".format(
                             len(vocabulary), len(idf_weights)))
      idf_weights = self._convert_to_ndarray(idf_weights)
      if idf_weights.ndim != 1:
        raise ValueError(
            "TF-IDF data must be a 1-index array, but received {}".format(
                type(idf_weights)))

      # If the passed vocabulary has no special tokens, we need to pad the front
      # of idf_weights. We don't have real document frequencies for these tokens
      # so we will use an average of all idf_weights passed in as a reasonable
      # default.
      if found_special_tokens:
        front_padding = 0
        front_padding_value = 0
      else:
        front_padding = token_start
        front_padding_value = np.average(idf_weights)
      # If pad_to_max_tokens is true, and max_tokens is greater than our total
      # vocab size, we need to pad the back of idf_weights with zeros as well.
      back_padding_value = 0
      if self.pad_to_max_tokens and self.max_tokens is not None:
        back_padding = self.max_tokens - front_padding - len(idf_weights)
      else:
        back_padding = 0
      weights = np.pad(
          idf_weights, (front_padding, back_padding),
          "constant",
          constant_values=(front_padding_value, back_padding_value))
      weights = tf.convert_to_tensor(weights, dtype=backend.floatx())
      self.idf_weights.assign(weights)
      self.idf_weights_const = self.idf_weights.value()

  def update_state(self, data):
    if self._has_input_vocabulary:
      raise ValueError(
          "Cannot adapt {} layer after setting a static vocabulary via init "
          "argument or `set_vocabulary`.".format(self.__class__.__name__))

    data = self._standardize_inputs(data, self.dtype)
    if data.shape.rank == 0:
      data = tf.expand_dims(data, -1)
    if data.shape.rank == 1:
      data = tf.expand_dims(data, -1)

    tokens, counts = self._num_tokens(data)
    self.token_counts.insert(tokens, counts + self.token_counts.lookup(tokens))

    if self.output_mode == TF_IDF:
      # Dedupe each row of our dataset.
      deduped_doc_data = tf.map_fn(lambda x: tf.unique(x)[0], data)
      # Flatten and count tokens.
      tokens, doc_counts = self._num_tokens(deduped_doc_data)
      self.token_document_counts.insert(
          tokens, doc_counts + self.token_document_counts.lookup(tokens))
      if tf_utils.is_ragged(data):
        self.num_documents.assign_add(data.nrows())
      else:
        self.num_documents.assign_add(tf.shape(data, out_type=tf.int64)[0])

  def finalize_state(self):
    if self._has_input_vocabulary or tf.equal(self.token_counts.size(), 0):
      # Finalize idf_weights to a const for call even if we don't need to
      # compute a new vocabulary.
      if self.output_mode == TF_IDF:
        self.idf_weights_const = self.idf_weights.value()
      return

    # Remove special tokens from our counts.
    if self.mask_token is not None:
      self.token_counts.remove(
          tf.convert_to_tensor([self.mask_token], self.dtype))
    if self.oov_token is not None:
      self.token_counts.remove(
          tf.convert_to_tensor([self.oov_token], self.dtype))

    tokens, counts = self.token_counts.export()
    # To keep vocabs deterministic, we sort our tokens by count and break ties
    # by sorting the tokens themselves. Tensorflow has no ops for sorting
    # strings, so we need to use numpy for the sort.
    sorted_indices = np.lexsort((tokens.numpy(), counts.numpy()))[::-1]
    token_start = self._token_start_index()
    if self.max_tokens:
      max_learned_tokens = self.max_tokens - token_start
      sorted_indices = sorted_indices[:max_learned_tokens]
    tokens = tf.gather(tokens, sorted_indices)
    self.lookup_table = self._lookup_table_from_tokens(tokens)

    if self.output_mode == TF_IDF:
      token_document_counts = self.token_document_counts.lookup(tokens)
      idf_weights = self._inverse_document_frequency(token_document_counts,
                                                     self.num_documents)
      idf_weights = tf.cast(idf_weights, backend.floatx())
      # Pad the front of idf_weights with the average idf weight for OOV tokens.
      # We cannot compute the real idf weight of OOV in a single pass.
      idf_weights = tf.pad(
          idf_weights, [[self._token_start_index(), 0]],
          constant_values=tf.reduce_mean(idf_weights))
      self.idf_weights.assign(idf_weights)
      self.idf_weights_const = self.idf_weights.value()

    # We call this here to save memory, now that we've built our vocabulary, we
    # don't want to keep every token we've seen in separate lookup tables.
    self.reset_state()

  def reset_state(self):  # pylint: disable=method-hidden
    if self._has_input_vocabulary:
      return

    self.token_counts.remove(self.token_counts.export()[0])
    if self.output_mode == TF_IDF:
      self.token_document_counts.remove(self.token_document_counts.export()[0])
      self.num_documents.assign(0)

  def call(self, inputs):
    self._maybe_freeze_vocab_size()

    inputs = self._standardize_inputs(inputs, self._key_dtype)
    original_shape = inputs.shape
    # Some ops will not handle scalar input, so uprank to rank 1.
    if inputs.shape.rank == 0:
      inputs = self._expand_dims(inputs, -1)

    if tf_utils.is_sparse(inputs):
      lookups = tf.SparseTensor(inputs.indices,
                                self._lookup_dense(inputs.values),
                                inputs.dense_shape)
    elif tf_utils.is_ragged(inputs):
      lookups = tf.ragged.map_flat_values(self._lookup_dense, inputs)
    else:
      lookups = self._lookup_dense(inputs)

    if self.output_mode == INT:
      # If we received a scalar input, downrank back to a scalar.
      if original_shape.rank == 0:
        lookups = tf.squeeze(lookups, -1)
      return lookups

    # One hot will unprank only if the final output dimension is not already 1.
    if self.output_mode == ONE_HOT:
      if lookups.shape[-1] != 1:
        lookups = self._expand_dims(lookups, -1)

    # TODO(b/190445202): remove output rank restriction.
    if lookups.shape.rank > 2:
      raise ValueError(
          "Received input shape {}, which would result in output rank {}. "
          "Currently only outputs up to rank 2 are supported for "
          "`output_mode={}`.".format(original_shape, lookups.shape.rank,
                                     self.output_mode))

    binary_output = self.output_mode in (MULTI_HOT, ONE_HOT)
    if self.pad_to_max_tokens:
      out_depth = self.max_tokens
    else:
      out_depth = self._frozen_vocab_size
    if self.sparse:
      bincounts = category_encoding.sparse_bincount(lookups, out_depth,
                                                    binary_output)
    else:
      bincounts = category_encoding.dense_bincount(lookups, out_depth,
                                                   binary_output)

    if self.output_mode == TF_IDF:
      return tf.multiply(bincounts, self.idf_weights_const)

    return bincounts

  def _lookup_dense(self, inputs):
    """Lookup table values for a dense Tensor, handling masking and OOV."""
    # When executing eagerly and tracing keras.Inputs, do not call lookup. This
    # is critical for restoring SavedModel, which will first trace layer.call
    # and then attempt to restore the table. We need the table to be unitialized
    # for the restore to work, but calling the table unitialized would error.
    if tf.executing_eagerly() and backend.is_keras_tensor(inputs):
      lookups = tf.zeros_like(inputs, dtype=self._value_dtype)
    else:
      lookups = self.lookup_table.lookup(inputs)

    if self.mask_token is not None:
      mask_locations = tf.equal(inputs, self._mask_key)
      lookups = tf.where(mask_locations, self._mask_value, lookups)

    if self.invert:
      return lookups

    lookup_checks = []

    if self.num_oov_indices == 0:
      # If we have zero oov indices, we need to check for oov inputs.
      oov_indices = tf.where(tf.equal(lookups, -1))
      oov_inputs = tf.gather_nd(inputs, oov_indices)
      msg = tf.strings.format(
          "When `num_oov_indices=0` all inputs should be in vocabulary, "
          "found OOV values {}, consider setting `num_oov_indices=1`.",
          (oov_inputs,))
      assertion = tf.Assert(tf.equal(tf.size(oov_indices), 0), [msg])
      lookup_checks.append(assertion)
    elif self.num_oov_indices > 1:
      # If we have multiple oov indices, we need a further hashing step.
      if self._key_dtype.is_integer:
        oov_indices = tf.math.floormod(inputs, self.num_oov_indices)
      else:
        oov_indices = tf.strings.to_hash_bucket_fast(
            inputs, num_buckets=self.num_oov_indices)
      oov_indices = oov_indices + self._oov_start_index()
      oov_locations = tf.equal(lookups, self._default_value)
      lookups = tf.where(oov_locations, oov_indices, lookups)

    with tf.control_dependencies(lookup_checks):
      return tf.identity(lookups)

  def _encode_output(self, lookups):
    """Encode the lookup result to the final output depending on output_mode."""

  def _uninitialized_lookup_table(self):
    with tf.init_scope():
      initializer = NullInitializer(self._key_dtype, self._value_dtype)
      return tf.lookup.StaticHashTable(initializer, self._default_value)

  def _lookup_table_from_tokens(self, tokens):
    with tf.init_scope():
      token_start = self._token_start_index()
      token_end = token_start + tf.size(tokens)
      indices = tf.range(token_start, token_end, dtype=tf.int64)
      keys, values = (indices, tokens) if self.invert else (tokens, indices)
      initializer = tf.lookup.KeyValueTensorInitializer(keys, values,
                                                        self._key_dtype,
                                                        self._value_dtype)
      table = tf.lookup.StaticHashTable(initializer, self._default_value)
      if not tf.compat.v1.executing_eagerly_outside_functions():
        backend.get_session().run(initializer.initialize(table))
      return table

  def _lookup_table_from_file(self, filename):
    if self.invert:
      key_index = tf.lookup.TextFileIndex.LINE_NUMBER
      value_index = tf.lookup.TextFileIndex.WHOLE_LINE
    else:
      key_index = tf.lookup.TextFileIndex.WHOLE_LINE
      value_index = tf.lookup.TextFileIndex.LINE_NUMBER
    with tf.init_scope():
      initializer = tf.lookup.TextFileInitializer(
          filename=filename,
          key_dtype=self._key_dtype,
          key_index=key_index,
          value_dtype=self._value_dtype,
          value_index=value_index,
          value_index_offset=self._token_start_index())
      table = tf.lookup.StaticHashTable(initializer, self._default_value)
      if not tf.compat.v1.executing_eagerly_outside_functions():
        backend.get_session().run(initializer.initialize(table))
      return table

  def _standardize_inputs(self, inputs, dtype):
    if isinstance(inputs, (list, tuple, np.ndarray)):
      inputs = tf.convert_to_tensor(inputs)
    if inputs.dtype != dtype:
      inputs = tf.cast(inputs, dtype)
    return inputs

  def _convert_to_ndarray(self, x):
    return np.array(x) if isinstance(x, (list, tuple)) else x

  def _expand_dims(self, inputs, axis):
    if tf_utils.is_sparse(inputs):
      return tf.sparse.expand_dims(inputs, axis)
    else:
      return tf.expand_dims(inputs, axis)

  def _make_serializable(self, x):
    if tf.is_tensor(x):
      x = x.numpy()
    if isinstance(x, (np.ndarray)):
      x = x.tolist()
      x = list(x)
    return x

  def _oov_start_index(self):
    return 1 if self.mask_token is not None and self.output_mode == INT else 0

  def _token_start_index(self):
    return self._oov_start_index() + self.num_oov_indices

  def _maybe_freeze_vocab_size(self):
    if self.output_mode == INT or self.pad_to_max_tokens:
      return
    with tf.init_scope():
      if not tf.executing_eagerly():
        raise RuntimeError(
            "When using `output_mode={}` eager mode execution must be enabled."
            .format(self.output_mode))
      new_vocab_size = self.vocabulary_size()
    if new_vocab_size == self._token_start_index():
      raise RuntimeError(
          "When using `output_mode={}` and `pad_to_max_tokens=False`, you "
          "must set the layer's vocabulary before calling it. Either pass "
          "a `vocabulary` argument to the layer, or call `adapt` with some "
          "sample data.".format(self.output_mode))
    elif (self._frozen_vocab_size is not None and
          new_vocab_size != self._frozen_vocab_size):
      raise RuntimeError(
          "When using `output_mode={}` and `pad_to_max_tokens=False`, the "
          "vocabulary size cannot be changed after the layer is called. "
          "Vocab size is {}, new vocab size is {}".format(
              self.output_mode, self._frozen_vocab_size, new_vocab_size))
    self._frozen_vocab_size = new_vocab_size

  def _find_repeated_tokens(self, vocabulary):
    """Return all repeated tokens in a vocabulary."""
    vocabulary_set = set(vocabulary)
    if len(vocabulary) != len(vocabulary_set):
      return [
          item for item, count in collections.Counter(vocabulary).items()
          if count > 1
      ]
    else:
      return []

  def _num_tokens(self, data):
    """Count the number of tokens in a ragged, sparse or dense tensor."""
    if tf_utils.is_sparse(data):
      flat_values = data.values
    elif tf_utils.is_ragged(data):
      flat_values = data.flat_values
    else:
      flat_values = tf.reshape(data, [-1])
    tokens, _, counts = tf.unique_with_counts(flat_values, out_idx=tf.int64)
    return tokens, counts

  def _inverse_document_frequency(self, token_document_counts, num_documents):
    """Computes the inverse-document-frequency (IDF) component of "tf_idf".

    Uses the default weighting scheme described in
    https://en.wikipedia.org/wiki/Tf%E2%80%93idf.

    Args:
      token_document_counts: An array of the # of documents each token appears
        in.
      num_documents: An int representing the total number of documents

    Returns:
      An array of "inverse document frequency" weights.
    """
    return tf.math.log(1 + num_documents / (1 + token_document_counts))

  @property
  def _trackable_saved_model_saver(self):
    return layer_serialization.IndexLookupLayerSavedModelSaver(self)

  # Override points for IntegerLookup and StringLookup.
  def _tensor_vocab_to_numpy(self, vocabulary):
    """Converts a tensor vocabulary to a numpy vocabulary."""
    return vocabulary.numpy()
