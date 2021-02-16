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
"""Utilities for working with tf.lookup tables in Keras."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

import collections
import os
import numpy as np
from keras import backend as K
from keras.utils import tf_utils
from tensorflow.python.ops import lookup_ops


class TableHandler(object):
  """Wrapper object that holds a lookup table and provides accessors."""

  def __init__(self,
               table,
               oov_tokens=None,
               mask_token=None,
               use_v1_apis=False):
    self.table = table

    # If we are using V1 APIs, and the table has an initializer, we need to run
    # it. However, not all tables have initializers, so we try-except here.
    if use_v1_apis:
      try:
        K.get_session().run(self.table.initializer)
      except AttributeError:
        pass

    self.mutable = isinstance(table, lookup_ops.MutableHashTable)
    self.mask_token = mask_token

    self.use_v1_apis = use_v1_apis
    if oov_tokens is None:
      self.oov_tokens = oov_tokens
    else:
      if not isinstance(oov_tokens, (list, tuple, np.ndarray)):
        oov_tokens = [oov_tokens]
      self.oov_tokens = tf.cast(oov_tokens, table._value_dtype)  # pylint: disable=protected-access

  def data(self):
    keys, values = self.table.export()
    return (self._eval(keys), self._eval(values))

  def vocab_size(self):
    return self._eval(self.table.size())

  def clear(self):
    if not self.mutable:
      return RuntimeError("Unable to clear a statically-backed table.")

    keys, _ = self.table.export()
    self._run(self.table.remove(keys))

  def insert(self, keys, values):
    """Insert values into the backed table."""
    if not self.mutable:
      raise RuntimeError("Unable to insert into a statically-backed table.")

    if len(values) != len(keys):
      raise RuntimeError("Size mismatch between values and key arrays. "
                         "Keys had size %s, values had size %s." %
                         (len(keys), len(values)))
    keys = tf.convert_to_tensor(
        keys, dtype=self.table._key_dtype)  # pylint: disable=protected-access
    values = tf.convert_to_tensor(
        values, dtype=self.table._value_dtype)  # pylint: disable=protected-access
    if values.shape.ndims != 1:
      raise ValueError("`values` must be 1-dimensional, got an input with "
                       " %s dimensions." % values.shape.ndims)
    self._run(self.table.insert(keys, values))

  def _replace_oov_buckets(self, inputs, lookups):
    """Replace the default OOV value with one of the OOV bucket values."""
    if self.oov_tokens is None:
      return lookups

    num_oov_elements = self.oov_tokens.shape.num_elements()
    if inputs.dtype.is_integer:
      oov_indices = tf.math.floormod(inputs, num_oov_elements)
    else:
      oov_indices = tf.strings.to_hash_bucket_fast(
          inputs, num_buckets=num_oov_elements)

    oov_values = tf.compat.v1.gather(self.oov_tokens, oov_indices)
    oov_locations = tf.equal(lookups, self.table._default_value)  # pylint: disable=protected-access

    return tf.compat.v1.where(oov_locations, oov_values, lookups)

  def _lookup_and_mask(self, inputs):
    """Return a lookup with any location with the mask_token masked to 0."""
    lookups = self.table.lookup(inputs)
    # If we don't need to handle masking, return the lookup values directly.
    if self.mask_token is None:
      return lookups

    # If we do need to handle masking, increment all the lookup values by 1
    # to account for the mask value at location 0. This also increments the
    # OOV value, so replace that. (This is inefficient, but we can't adjust
    # the table safely, so we don't have a choice.)
    oov_locations = tf.equal(lookups, self.table._default_value)  # pylint: disable=protected-access
    oov_values = tf.compat.v1.ones_like(
        lookups, dtype=self.table._value_dtype) * self.table._default_value  # pylint: disable=protected-access
    adjusted_lookups = tf.compat.v1.where(oov_locations, oov_values, lookups)

    # Inject 0s wherever the mask token was in the inputs.
    mask_locations = tf.equal(inputs, self.mask_token)
    return tf.compat.v1.where(
        mask_locations,
        tf.compat.v1.zeros_like(lookups, dtype=self.table._value_dtype),  # pylint: disable=protected-access
        adjusted_lookups)  # pylint: disable=protected-access

  def _ragged_lookup(self, inputs):
    """Perform a table lookup on a ragged tensor."""
    # The table lookup ops don't natively support ragged tensors, so if we have
    # a RT we need to use map_flat_values to look up every element.
    indexed_data = tf.ragged.map_flat_values(
        self._lookup_and_mask, inputs)
    indexed_data = tf.ragged.map_flat_values(
        self._replace_oov_buckets, inputs, indexed_data)
    # table.lookup is not shape-preserving, so we need to set the shape here.
    indexed_data._set_shape(inputs.shape)  # pylint: disable=protected-access
    # Composite tensors can pass tensor values through, which will cause
    # errors if all operations in the TF graph do so. We can break this chain
    # with an identity here.
    return tf.identity(indexed_data)

  def _sparse_lookup(self, inputs):
    """Perform a table lookup on a sparse tensor."""
    values = self._lookup_and_mask(inputs.values)
    values = self._replace_oov_buckets(inputs.values, values)
    indexed_data = tf.SparseTensor(inputs.indices, values,
                                              inputs.dense_shape)
    # Composite tensors can pass tensor values through, which will cause
    # errors if all operations in the TF graph do so. We can break this chain
    # with an identity here.
    return tf.identity(indexed_data)

  def _tensor_lookup(self, inputs):
    """Perform a table lookup on a tf.tensor."""
    values = self._lookup_and_mask(inputs)
    indexed_data = self._replace_oov_buckets(inputs, values)
    # (b/149446477): output does not preserve input shape.
    indexed_data.set_shape(inputs.shape)
    return indexed_data

  def lookup(self, inputs):
    """Perform a table lookup."""
    # Sparse tensors don't play nicely with tensor conversion, so we handle
    # them before attempting to convert lists or arrays to tensors.
    if isinstance(
        inputs, (tf.SparseTensor, tf.compat.v1.SparseTensorValue)):
      return self._sparse_lookup(inputs)

    if tf_utils.is_ragged(inputs):
      if isinstance(inputs, tf.compat.v1.ragged.RaggedTensorValue):
        flat_values = tf.convert_to_tensor(
            value=inputs.flat_values,
            name="flat_values")
        inputs = tf.RaggedTensor.from_nested_row_splits(
            flat_values, inputs.nested_row_splits, validate=False)
      return self._ragged_lookup(inputs)

    # For normal tensor inputs
    inputs = tf.convert_to_tensor(inputs)
    return self._tensor_lookup(inputs)

  def _eval(self, tensor):
    if self.use_v1_apis:
      return K.get_session().run(tensor)
    else:
      return tensor.numpy()

  def _run(self, op):
    if self.use_v1_apis:
      K.get_session().run(op)


def get_vocabulary_from_file(vocabulary_path, encoding="utf-8"):
  """Read a vocabulary in from a file."""
  vocab = []
  with tf.io.gfile.GFile(vocabulary_path, "r") as reader:
    while True:
      # Get the next line (incl. \n), and break if nothing is left to read.
      text = reader.readline()
      if not text:
        break

      # Convert the raw text and strip whitespace.
      if isinstance(text, str):
        token = text
      elif isinstance(text, bytes):
        token = text.decode(encoding, "ignore")
      token = token.rstrip(os.linesep)
      vocab.append(token)
  return vocab


def validate_vocabulary_is_unique(vocabulary):
  """Validate that a vocabulary contains no repeated tokens."""
  vocabulary_set = set(vocabulary)
  if len(vocabulary) != len(vocabulary_set):
    repeated_items = [
        item for item, count in collections.Counter(vocabulary).items()
        if count > 1
    ]
    raise ValueError("The passed vocabulary has at least one repeated "
                     "term. Please uniquify your dataset. The repeated terms "
                     "are %s" % repeated_items)


def assert_same_type(expected_type, values, value_name):
  """Assert that 'values' is of type 'expected_type'."""
  if tf.as_dtype(expected_type) != tf.as_dtype(values.dtype):
    raise RuntimeError("Expected %s type %s, got %s" %
                       (value_name, expected_type, values.dtype))


def convert_to_ndarray(x, dtype=None):
  """Convert 'x' to a numpy array."""
  array = np.array(x) if isinstance(x, (list, tuple)) else x
  if dtype not in (None, tf.string):
    # If the dtype is an integer, we do permissive casting. This allows
    # users to examine int32 data if the dtype is int64 without trouble.
    np_dtype = tf.as_dtype(dtype).as_numpy_dtype
    if np.can_cast(array.dtype, np_dtype):
      array = array.astype(np_dtype, casting="safe")
  return array
