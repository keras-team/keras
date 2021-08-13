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
"""Keras discretization preprocessing layer."""

# pylint: disable=g-classes-have-attributes
# pylint: disable=g-direct-tensorflow-import

from keras.engine import base_preprocessing_layer
from keras.layers.preprocessing import preprocessing_utils as utils
from keras.utils import tf_utils
import numpy as np
import tensorflow.compat.v2 as tf
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util.tf_export import keras_export


def summarize(values, epsilon):
  """Reduce a 1D sequence of values to a summary.

  This algorithm is based on numpy.quantiles but modified to allow for
  intermediate steps between multiple data sets. It first finds the target
  number of bins as the reciprocal of epsilon and then takes the individual
  values spaced at appropriate intervals to arrive at that target.
  The final step is to return the corresponding counts between those values
  If the target num_bins is larger than the size of values, the whole array is
  returned (with weights of 1).

  Args:
      values: 1D `np.ndarray` to be summarized.
      epsilon: A `'float32'` that determines the approximate desired precision.

  Returns:
      A 2D `np.ndarray` that is a summary of the inputs. First column is the
      interpolated partition values, the second is the weights (counts).
  """

  values = tf.reshape(values, [-1])
  values = tf.sort(values)
  elements = tf.cast(tf.size(values), tf.float32)
  num_buckets = 1. / epsilon
  increment = tf.cast(elements / num_buckets, tf.int32)
  start = increment
  step = tf.maximum(increment, 1)
  boundaries = values[start::step]
  weights = tf.ones_like(boundaries)
  weights = weights * tf.cast(step, tf.float32)
  return tf.stack([boundaries, weights])


def compress(summary, epsilon):
  """Compress a summary to within `epsilon` accuracy.

  The compression step is needed to keep the summary sizes small after merging,
  and also used to return the final target boundaries. It finds the new bins
  based on interpolating cumulative weight percentages from the large summary.
  Taking the difference of the cumulative weights from the previous bin's
  cumulative weight will give the new weight for that bin.

  Args:
      summary: 2D `np.ndarray` summary to be compressed.
      epsilon: A `'float32'` that determines the approxmiate desired precision.

  Returns:
      A 2D `np.ndarray` that is a compressed summary. First column is the
      interpolated partition values, the second is the weights (counts).
  """
  # TODO(b/184863356): remove the numpy escape hatch here.
  return tf.numpy_function(
      lambda s: _compress_summary_numpy(s, epsilon), [summary], tf.float32)


def _compress_summary_numpy(summary, epsilon):
  """Compress a summary with numpy."""
  if summary.shape[1] * epsilon < 1:
    return summary

  percents = epsilon + np.arange(0.0, 1.0, epsilon)
  cum_weights = summary[1].cumsum()
  cum_weight_percents = cum_weights / cum_weights[-1]
  new_bins = np.interp(percents, cum_weight_percents, summary[0])
  cum_weights = np.interp(percents, cum_weight_percents, cum_weights)
  new_weights = cum_weights - np.concatenate((np.array([0]), cum_weights[:-1]))
  summary = np.stack((new_bins, new_weights))
  return summary.astype(np.float32)


def merge_summaries(prev_summary, next_summary, epsilon):
  """Weighted merge sort of summaries.

  Given two summaries of distinct data, this function merges (and compresses)
  them to stay within `epsilon` error tolerance.

  Args:
      prev_summary: 2D `np.ndarray` summary to be merged with `next_summary`.
      next_summary: 2D `np.ndarray` summary to be merged with `prev_summary`.
      epsilon: A float that determines the approxmiate desired precision.

  Returns:
      A 2-D `np.ndarray` that is a merged summary. First column is the
      interpolated partition values, the second is the weights (counts).
  """
  merged = tf.concat((prev_summary, next_summary), axis=1)
  merged = tf.gather(merged, tf.argsort(merged[0]), axis=1)
  return compress(merged, epsilon)


def get_bin_boundaries(summary, num_bins):
  return compress(summary, 1.0 / num_bins)[0, :-1]


@keras_export("keras.layers.Discretization",
              "keras.layers.experimental.preprocessing.Discretization")
class Discretization(base_preprocessing_layer.PreprocessingLayer):
  """Buckets data into discrete ranges.

  This layer will place each element of its input data into one of several
  contiguous ranges and output an integer index indicating which range each
  element was placed in.

  Input shape:
    Any `tf.Tensor` or `tf.RaggedTensor` of dimension 2 or higher.

  Output shape:
    Same as input shape.

  Attributes:
    bin_boundaries: A list of bin boundaries. The leftmost and rightmost bins
      will always extend to `-inf` and `inf`, so `bin_boundaries=[0., 1., 2.]`
      generates bins `(-inf, 0.)`, `[0., 1.)`, `[1., 2.)`, and `[2., +inf)`. If
      this option is set, `adapt` should not be called.
    num_bins: The integer number of bins to compute. If this option is set,
      `adapt` should be called to learn the bin boundaries.
    epsilon: Error tolerance, typically a small fraction close to zero (e.g.
      0.01). Higher values of epsilon increase the quantile approximation, and
      hence result in more unequal buckets, but could improve performance
      and resource consumption.

  Examples:

  Bucketize float values based on provided buckets.
  >>> input = np.array([[-1.5, 1.0, 3.4, .5], [0.0, 3.0, 1.3, 0.0]])
  >>> layer = tf.keras.layers.Discretization(bin_boundaries=[0., 1., 2.])
  >>> layer(input)
  <tf.Tensor: shape=(2, 4), dtype=int64, numpy=
  array([[0, 2, 3, 1],
         [1, 3, 2, 1]], dtype=int64)>

  Bucketize float values based on a number of buckets to compute.
  >>> input = np.array([[-1.5, 1.0, 3.4, .5], [0.0, 3.0, 1.3, 0.0]])
  >>> layer = tf.keras.layers.Discretization(num_bins=4, epsilon=0.01)
  >>> layer.adapt(input)
  >>> layer(input)
  <tf.Tensor: shape=(2, 4), dtype=int64, numpy=
  array([[0, 2, 3, 2],
         [1, 3, 3, 1]], dtype=int64)>
  """

  def __init__(self,
               bin_boundaries=None,
               num_bins=None,
               epsilon=0.01,
               **kwargs):
    # bins is a deprecated arg for setting bin_boundaries or num_bins that still
    # has some usage.
    if "bins" in kwargs:
      logging.warning(
          "bins is deprecated, please use bin_boundaries or num_bins instead.")
      if isinstance(kwargs["bins"], int) and num_bins is None:
        num_bins = kwargs["bins"]
      elif bin_boundaries is None:
        bin_boundaries = kwargs["bins"]
      del kwargs["bins"]
    super().__init__(**kwargs)
    base_preprocessing_layer.keras_kpl_gauge.get_cell("Discretization").set(
        True)
    if num_bins is not None and num_bins < 0:
      raise ValueError("`num_bins` must be greater than or equal to 0. "
                       "You passed `num_bins={}`".format(num_bins))
    if num_bins is not None and bin_boundaries is not None:
      raise ValueError("Both `num_bins` and `bin_boundaries` should not be "
                       "set. You passed `num_bins={}` and "
                       "`bin_boundaries={}`".format(num_bins, bin_boundaries))
    bin_boundaries = utils.listify_tensors(bin_boundaries)
    self.input_bin_boundaries = bin_boundaries
    self.bin_boundaries = bin_boundaries if bin_boundaries is not None else []
    self.num_bins = num_bins
    self.epsilon = epsilon

  def build(self, input_shape):
    super().build(input_shape)

    if self.input_bin_boundaries is not None:
      return

    # Summary contains two equal length vectors of bins at index 0 and weights
    # at index 1.
    self.summary = self.add_weight(
        name="summary",
        shape=(2, None),
        dtype=tf.float32,
        initializer=lambda shape, dtype: [[], []],  # pylint: disable=unused-arguments
        trainable=False)

  def update_state(self, data):
    if self.input_bin_boundaries is not None:
      raise ValueError(
          "Cannot adapt a Discretization layer that has been initialized with "
          "`bin_boundaries`, use `num_bins` instead. You passed "
          "`bin_boundaries={}`.".format(self.input_bin_boundaries))

    if not self.built:
      raise RuntimeError("`build` must be called before `update_state`.")

    data = tf.convert_to_tensor(data)
    if data.dtype != tf.float32:
      data = tf.cast(data, tf.float32)
    summary = summarize(data, self.epsilon)
    self.summary.assign(merge_summaries(summary, self.summary, self.epsilon))

  def finalize_state(self):
    if self.input_bin_boundaries is not None or not self.built:
      return

    # The bucketize op only support list boundaries.
    self.bin_boundaries = utils.listify_tensors(
        get_bin_boundaries(self.summary, self.num_bins))

  def reset_state(self):  # pylint: disable=method-hidden
    if self.input_bin_boundaries is not None or not self.built:
      return

    self.summary.assign([[], []])

  def get_config(self):
    config = super().get_config()
    config.update({
        "bin_boundaries": self.input_bin_boundaries,
        "num_bins": self.num_bins,
        "epsilon": self.epsilon,
    })
    return config

  def compute_output_shape(self, input_shape):
    return input_shape

  def compute_output_signature(self, input_spec):
    output_shape = self.compute_output_shape(input_spec.shape.as_list())
    output_dtype = tf.int64
    if isinstance(input_spec, tf.SparseTensorSpec):
      return tf.SparseTensorSpec(
          shape=output_shape, dtype=output_dtype)
    return tf.TensorSpec(shape=output_shape, dtype=output_dtype)

  def call(self, inputs):
    def bucketize(inputs):
      outputs = tf.raw_ops.Bucketize(
          input=inputs, boundaries=self.bin_boundaries)
      # All other preprocessing layers use int64 for int output, so we conform
      # here. Sadly the underlying op only supports int32, so we need to cast.
      return tf.cast(outputs, tf.int64)

    if tf_utils.is_ragged(inputs):
      integer_buckets = tf.ragged.map_flat_values(bucketize, inputs)
      # Ragged map_flat_values doesn't touch the non-values tensors in the
      # ragged composite tensor. If this op is the only op a Keras model,
      # this can cause errors in Graph mode, so wrap the tensor in an identity.
      return tf.identity(integer_buckets)
    elif tf_utils.is_sparse(inputs):
      return tf.SparseTensor(
          indices=tf.identity(inputs.indices),
          values=bucketize(inputs.values),
          dense_shape=tf.identity(inputs.dense_shape))
    else:
      return bucketize(inputs)
