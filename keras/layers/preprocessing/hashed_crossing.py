# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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
"""Keras hashed crossing preprocessing layer."""

# pylint: disable=g-classes-have-attributes
# pylint: disable=g-direct-tensorflow-import

from keras import backend
from keras.engine import base_layer
from keras.engine import base_preprocessing_layer
from keras.layers.preprocessing import preprocessing_utils as utils
from keras.utils import layer_utils
import tensorflow.compat.v2 as tf
from tensorflow.python.util.tf_export import keras_export

INT = utils.INT
ONE_HOT = utils.ONE_HOT


@keras_export("keras.layers.experimental.preprocessing.HashedCrossing")
class HashedCrossing(base_layer.Layer):
  """A preprocessing layer which crosses features using the "hashing trick".

  This layer performs crosses of categorical features using the "hasing trick".
  Conceptually, the transformation can be thought of as:
  hash(concatenation of features) % `num_bins`.

  This layer currently only performs crosses of scalar inputs and batches of
  scalar inputs. Valid input shapes are `(batch_size, 1)`, `(batch_size,)` and
  `()`.

  For an overview and full list of preprocessing layers, see the preprocessing
  [guide](https://www.tensorflow.org/guide/keras/preprocessing_layers).

  Args:
    num_bins: Number of hash bins.
    output_mode: Specification for the output of the layer. Defaults to `"int"`.
      Values can be `"int"`, or `"one_hot"` configuring the layer as follows:
        - `"int"`: Return the integer bin indices directly.
        - `"one_hot"`: Encodes each individual element in the input into an
          array the same size as `num_bins`, containing a 1 at the input's bin
          index.
    sparse: Boolean. Only applicable to `"one_hot"` mode. If True, returns a
      `SparseTensor` instead of a dense `Tensor`. Defaults to False.
    **kwargs: Keyword arguments to construct a layer.

  Examples:

  **Crossing two scalar features.**

  >>> layer = tf.keras.layers.experimental.preprocessing.HashedCrossing(
  ...     num_bins=5)
  >>> feat1 = tf.constant(['A', 'B', 'A', 'B', 'A'])
  >>> feat2 = tf.constant([101, 101, 101, 102, 102])
  >>> layer((feat1, feat2))
  <tf.Tensor: shape=(5,), dtype=int64, numpy=array([1, 4, 1, 1, 3])>

  **Crossing and one-hotting two scalar features.**

  >>> layer = tf.keras.layers.experimental.preprocessing.HashedCrossing(
  ...     num_bins=5, output_mode='one_hot')
  >>> feat1 = tf.constant(['A', 'B', 'A', 'B', 'A'])
  >>> feat2 = tf.constant([101, 101, 101, 102, 102])
  >>> layer((feat1, feat2))
  <tf.Tensor: shape=(5, 5), dtype=float32, numpy=
    array([[0., 1., 0., 0., 0.],
           [0., 0., 0., 0., 1.],
           [0., 1., 0., 0., 0.],
           [0., 1., 0., 0., 0.],
           [0., 0., 0., 1., 0.]], dtype=float32)>
  """

  def __init__(self,
               num_bins,
               output_mode="int",
               sparse=False,
               **kwargs):
    # By default, output int64 when output_mode="int" and floats otherwise.
    if "dtype" not in kwargs or kwargs["dtype"] is None:
      kwargs["dtype"] = tf.int64 if output_mode == INT else backend.floatx()

    super().__init__(**kwargs)
    base_preprocessing_layer.keras_kpl_gauge.get_cell(
        "HashedCrossing").set(True)

    # Check dtype only after base layer parses it; dtype parsing is complex.
    if output_mode == INT and not tf.as_dtype(self.compute_dtype).is_integer:
      input_dtype = kwargs["dtype"]
      raise ValueError("When `output_mode='int'`, `dtype` should be an integer "
                       f"type. Received: dtype={input_dtype}")

    # "output_mode" must be one of (INT, ONE_HOT)
    layer_utils.validate_string_arg(
        output_mode,
        allowable_strings=(INT, ONE_HOT),
        layer_name=self.__class__.__name__,
        arg_name="output_mode")

    self.num_bins = num_bins
    self.output_mode = output_mode
    self.sparse = sparse

  def call(self, inputs):
    # Convert all inputs to tensors and check shape. This layer only supports
    # sclars and batches of scalars for the initial version.
    self._check_at_least_two_inputs(inputs)
    inputs = [utils.ensure_tensor(x) for x in inputs]
    self._check_input_shape_and_type(inputs)

    # Uprank to rank 2 for the cross_hashed op.
    rank = inputs[0].shape.rank
    if rank < 2:
      inputs = [utils.expand_dims(x, -1) for x in inputs]
    if rank < 1:
      inputs = [utils.expand_dims(x, -1) for x in inputs]

    # Perform the cross and convert to dense
    outputs = tf.sparse.cross_hashed(inputs, self.num_bins)
    outputs = tf.sparse.to_dense(outputs)

    # Fix output shape and downrank to match input rank.
    if rank == 2:
      # tf.sparse.cross_hashed output shape will always be None on the last
      # dimension. Given our input shape restrictions, we want to force shape 1
      # instead.
      outputs = tf.reshape(outputs, [-1, 1])
    elif rank == 1:
      outputs = tf.reshape(outputs, [-1])
    elif rank == 0:
      outputs = tf.reshape(outputs, [])

    # Encode outputs.
    return utils.encode_categorical_inputs(
        outputs,
        output_mode=self.output_mode,
        depth=self.num_bins,
        sparse=self.sparse,
        dtype=self.compute_dtype)

  def compute_output_shape(self, input_shapes):
    self._check_at_least_two_inputs(input_shapes)
    return utils.compute_shape_for_encode_categorical(input_shapes[0])

  def compute_output_signature(self, input_specs):
    input_shapes = [x.shape.as_list() for x in input_specs]
    output_shape = self.compute_output_shape(input_shapes)
    if self.sparse or any(
        isinstance(x, tf.SparseTensorSpec) for x in input_specs):
      return tf.SparseTensorSpec(shape=output_shape, dtype=self.compute_dtype)
    return tf.TensorSpec(shape=output_shape, dtype=self.compute_dtype)

  def get_config(self):
    config = super().get_config()
    config.update({
        "num_bins": self.num_bins,
        "output_mode": self.output_mode,
        "sparse": self.sparse,
    })
    return config

  def _check_at_least_two_inputs(self, inputs):
    if not isinstance(inputs, (list, tuple)):
      raise ValueError(
          "`HashedCrossing` should be called on a list or tuple of inputs. "
          f"Received: inputs={inputs}")
    if len(inputs) < 2:
      raise ValueError(
          "`HashedCrossing` should be called on at least two inputs. "
          f"Received: inputs={inputs}")

  def _check_input_shape_and_type(self, inputs):
    first_shape = inputs[0].shape.as_list()
    rank = len(first_shape)
    if rank > 2 or (rank == 2 and first_shape[-1] != 1):
      raise ValueError(
          "All `HashedCrossing` inputs should have shape `[]`, `[batch_size]` "
          f"or `[batch_size, 1]`. Received: inputs={inputs}")
    if not all(x.shape.as_list() == first_shape for x in inputs[1:]):
      raise ValueError("All `HashedCrossing` inputs should have equal shape. "
                       f"Received: inputs={inputs}")
    if any(isinstance(x, (tf.RaggedTensor, tf.SparseTensor)) for x in inputs):
      raise ValueError("All `HashedCrossing` inputs should be dense tensors. "
                       f"Received: inputs={inputs}")
    if not all(x.dtype.is_integer or x.dtype == tf.string for x in inputs):
      raise ValueError("All `HashedCrossing` inputs should have an integer or "
                       f"string dtype. Received: inputs={inputs}")
