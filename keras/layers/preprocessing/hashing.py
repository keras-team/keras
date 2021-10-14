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
"""Keras hashing preprocessing layer."""

# pylint: disable=g-classes-have-attributes
# pylint: disable=g-direct-tensorflow-import

from keras import backend
from keras.engine import base_layer
from keras.engine import base_preprocessing_layer
from keras.layers.preprocessing import preprocessing_utils as utils
from keras.utils import layer_utils
import numpy as np
import tensorflow.compat.v2 as tf
from tensorflow.python.util.tf_export import keras_export

INT = utils.INT
MULTI_HOT = utils.MULTI_HOT
ONE_HOT = utils.ONE_HOT
COUNT = utils.COUNT


@keras_export('keras.layers.Hashing',
              'keras.layers.experimental.preprocessing.Hashing')
class Hashing(base_layer.Layer):
  """A preprocessing layer which hashes and bins categorical features.

  This layer transforms categorical inputs to hashed output. It element-wise
  converts a ints or strings to ints in a fixed range. The stable hash
  function uses `tensorflow::ops::Fingerprint` to produce the same output
  consistently across all platforms.

  This layer uses [FarmHash64](https://github.com/google/farmhash) by default,
  which provides a consistent hashed output across different platforms and is
  stable across invocations, regardless of device and context, by mixing the
  input bits thoroughly.

  If you want to obfuscate the hashed output, you can also pass a random `salt`
  argument in the constructor. In that case, the layer will use the
  [SipHash64](https://github.com/google/highwayhash) hash function, with
  the `salt` value serving as additional input to the hash function.

  For an overview and full list of preprocessing layers, see the preprocessing
  [guide](https://www.tensorflow.org/guide/keras/preprocessing_layers).

  **Example (FarmHash64)**

  >>> layer = tf.keras.layers.Hashing(num_bins=3)
  >>> inp = [['A'], ['B'], ['C'], ['D'], ['E']]
  >>> layer(inp)
  <tf.Tensor: shape=(5, 1), dtype=int64, numpy=
    array([[1],
           [0],
           [1],
           [1],
           [2]])>

  **Example (FarmHash64) with a mask value**

  >>> layer = tf.keras.layers.Hashing(num_bins=3, mask_value='')
  >>> inp = [['A'], ['B'], [''], ['C'], ['D']]
  >>> layer(inp)
  <tf.Tensor: shape=(5, 1), dtype=int64, numpy=
    array([[1],
           [1],
           [0],
           [2],
           [2]])>

  **Example (SipHash64)**

  >>> layer = tf.keras.layers.Hashing(num_bins=3, salt=[133, 137])
  >>> inp = [['A'], ['B'], ['C'], ['D'], ['E']]
  >>> layer(inp)
  <tf.Tensor: shape=(5, 1), dtype=int64, numpy=
    array([[1],
           [2],
           [1],
           [0],
           [2]])>

  **Example (Siphash64 with a single integer, same as `salt=[133, 133]`)**

  >>> layer = tf.keras.layers.Hashing(num_bins=3, salt=133)
  >>> inp = [['A'], ['B'], ['C'], ['D'], ['E']]
  >>> layer(inp)
  <tf.Tensor: shape=(5, 1), dtype=int64, numpy=
    array([[0],
           [0],
           [2],
           [1],
           [0]])>

  Args:
    num_bins: Number of hash bins. Note that this includes the `mask_value` bin,
      so the effective number of bins is `(num_bins - 1)` if `mask_value` is
      set.
    mask_value: A value that represents masked inputs, which are mapped to
      index 0. Defaults to None, meaning no mask term will be added and the
      hashing will start at index 0.
    salt: A single unsigned integer or None.
      If passed, the hash function used will be SipHash64, with these values
      used as an additional input (known as a "salt" in cryptography).
      These should be non-zero. Defaults to `None` (in that
      case, the FarmHash64 hash function is used). It also supports
      tuple/list of 2 unsigned integer numbers, see reference paper for details.
    output_mode: Specification for the output of the layer. Defaults to `"int"`.
      Values can be `"int"`, `"one_hot"`, `"multi_hot"`, or `"count"`
      configuring the layer as follows:
        - `"int"`: Return the integer bin indices directly.
        - `"one_hot"`: Encodes each individual element in the input into an
          array the same size as `num_bins`, containing a 1 at the input's bin
          index. If the last dimension is size 1, will encode on that dimension.
          If the last dimension is not size 1, will append a new dimension for
          the encoded output.
        - `"multi_hot"`: Encodes each sample in the input into a single array
          the same size as `num_bins`, containing a 1 for each bin index
          index present in the sample. Treats the last dimension as the sample
          dimension, if input shape is `(..., sample_length)`, output shape will
          be `(..., num_tokens)`.
        - `"count"`: As `"multi_hot"`, but the int array contains a count of the
          number of times the bin index appeared in the sample.
    sparse: Boolean. Only applicable to `"one_hot"`, `"multi_hot"`,
      and `"count"` output modes. If True, returns a `SparseTensor` instead of
      a dense `Tensor`. Defaults to False.
    **kwargs: Keyword arguments to construct a layer.

  Input shape:
    A single or list of string, int32 or int64 `Tensor`,
    `SparseTensor` or `RaggedTensor` of shape `(batch_size, ...,)`

  Output shape:
    An int64 `Tensor`, `SparseTensor` or `RaggedTensor` of shape
    `(batch_size, ...)`. If any input is `RaggedTensor` then output is
    `RaggedTensor`, otherwise if any input is `SparseTensor` then output is
    `SparseTensor`, otherwise the output is `Tensor`.

  Reference:
    - [SipHash with salt](https://www.131002.net/siphash/siphash.pdf)

  """

  def __init__(self,
               num_bins,
               mask_value=None,
               salt=None,
               output_mode='int',
               sparse=False,
               **kwargs):
    if num_bins is None or num_bins <= 0:
      raise ValueError(
          f'The `num_bins` for `Hashing` cannot be `None` or non-positive '
          f'values. Received: num_bins={num_bins}.')

    # By default, output int64 when output_mode='int' and floats otherwise.
    if 'dtype' not in kwargs or kwargs['dtype'] is None:
      kwargs['dtype'] = tf.int64 if output_mode == INT else backend.floatx()
    elif output_mode == 'int' and not tf.as_dtype(kwargs['dtype']).is_integer:
      # Compat for when dtype was alwyas floating and ingored by the layer.
      kwargs['dtype'] = tf.int64

    super().__init__(**kwargs)
    base_preprocessing_layer.keras_kpl_gauge.get_cell('Hashing').set(True)

    # Check dtype only after base layer parses it; dtype parsing is complex.
    if output_mode == INT and not tf.as_dtype(self.compute_dtype).is_integer:
      input_dtype = kwargs['dtype']
      raise ValueError('When `output_mode="int"`, `dtype` should be an integer '
                       f'type. Received: dtype={input_dtype}')

    # 'output_mode' must be one of (INT, ONE_HOT, MULTI_HOT, COUNT)
    layer_utils.validate_string_arg(
        output_mode,
        allowable_strings=(INT, ONE_HOT, MULTI_HOT, COUNT),
        layer_name=self.__class__.__name__,
        arg_name='output_mode')

    if sparse and output_mode == INT:
      raise ValueError(f'`sparse` may only be true if `output_mode` is '
                       f'`"one_hot"`, `"multi_hot"`, or `"count"`. '
                       f'Received: sparse={sparse} and '
                       f'output_mode={output_mode}')

    self.num_bins = num_bins
    self.mask_value = mask_value
    self.strong_hash = True if salt is not None else False
    self.output_mode = output_mode
    self.sparse = sparse
    self.salt = None
    if salt is not None:
      if isinstance(salt, (tuple, list)) and len(salt) == 2:
        self.salt = salt
      elif isinstance(salt, int):
        self.salt = [salt, salt]
      else:
        raise ValueError(
            f'The `salt` argument for `Hashing` can only be a tuple of size 2 '
            f'integers, or a single integer. Received: salt={salt}.')

  def call(self, inputs):
    if isinstance(inputs, (list, tuple, np.ndarray)):
      inputs = tf.convert_to_tensor(inputs)
    if isinstance(inputs, tf.SparseTensor):
      indices = tf.SparseTensor(
          indices=inputs.indices,
          values=self._hash_values_to_bins(inputs.values),
          dense_shape=inputs.dense_shape)
    else:
      indices = self._hash_values_to_bins(inputs)
    return utils.encode_categorical_inputs(
        indices,
        output_mode=self.output_mode,
        depth=self.num_bins,
        sparse=self.sparse,
        dtype=self.compute_dtype)

  def _hash_values_to_bins(self, values):
    """Converts a non-sparse tensor of values to bin indices."""
    hash_bins = self.num_bins
    mask = None
    # If mask_value is set, the zeroth bin is reserved for it.
    if self.mask_value is not None and hash_bins > 1:
      hash_bins -= 1
      mask = tf.equal(values, self.mask_value)
    # Convert all values to strings before hashing.
    if values.dtype.is_integer:
      values = tf.as_string(values)
    # Hash the strings.
    if self.strong_hash:
      values = tf.strings.to_hash_bucket_strong(
          values, hash_bins, name='hash', key=self.salt)
    else:
      values = tf.strings.to_hash_bucket_fast(values, hash_bins, name='hash')
    if mask is not None:
      values = tf.add(values, tf.ones_like(values))
      values = tf.where(mask, tf.zeros_like(values), values)
    return values

  def compute_output_shape(self, input_shape):
    return input_shape

  def compute_output_signature(self, input_spec):
    output_shape = self.compute_output_shape(input_spec.shape)
    if isinstance(input_spec, tf.SparseTensorSpec):
      return tf.SparseTensorSpec(shape=output_shape, dtype=self.compute_dtype)
    else:
      return tf.TensorSpec(shape=output_shape, dtype=self.compute_dtype)

  def get_config(self):
    config = super().get_config()
    config.update({
        'num_bins': self.num_bins,
        'salt': self.salt,
        'mask_value': self.mask_value,
        'output_mode': self.output_mode,
        'sparse': self.sparse,
    })
    return config
