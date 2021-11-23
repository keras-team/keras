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

from keras import backend
from keras.engine import base_preprocessing_layer
from keras.layers.preprocessing import preprocessing_utils as utils
from keras.utils import layer_utils
from keras.utils import tf_utils
import numpy as np
import tensorflow.compat.v2 as tf
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util.tf_export import keras_export

INT = utils.INT
MULTI_HOT = utils.MULTI_HOT
ONE_HOT = utils.ONE_HOT
COUNT = utils.COUNT


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
  """A preprocessing layer which buckets continuous features by ranges.

  This layer will place each element of its input data into one of several
  contiguous ranges and output an integer index indicating which range each
  element was placed in.

  For an overview and full list of preprocessing layers, see the preprocessing
  [guide](https://www.tensorflow.org/guide/keras/preprocessing_layers).

  Input shape:
    Any `tf.Tensor` or `tf.RaggedTensor` of dimension 2 or higher.

  Output shape:
    Same as input shape.

  Attributes:
    bin_boundaries: A list of bin boundaries. The leftmost and rightmost bins
      will always extend to `-inf` and `inf`, so `bin_boundaries=[0., 1., 2.]`
      generates bins `(-inf, 0.)`, `[0., 1.)`, `[1., 2.)`, and `[2., +inf)`. If
      this option is set, `adapt()` should not be called.
    num_bins: The integer number of bins to compute. If this option is set,
      `adapt()` should be called to learn the bin boundaries.
    epsilon: Error tolerance, typically a small fraction close to zero (e.g.
      0.01). Higher values of epsilon increase the quantile approximation, and
      hence result in more unequal buckets, but could improve performance
      and resource consumption.
    output_mode: Specification for the output of the layer. Defaults to `"int"`.
      Values can be `"int"`, `"one_hot"`, `"multi_hot"`, or `"count"`
      configuring the layer as follows:
        - `"int"`: Return the discritized bin indices directly.
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
               output_mode="int",
               sparse=False,
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

    # By default, output int64 when output_mode='int' and floats otherwise.
    if "dtype" not in kwargs or kwargs["dtype"] is None:
      kwargs["dtype"] = tf.int64 if output_mode == INT else backend.floatx()
    elif output_mode == "int" and not tf.as_dtype(kwargs["dtype"]).is_integer:
      # Compat for when dtype was always floating and ignored by the layer.
      kwargs["dtype"] = tf.int64

    super().__init__(**kwargs)
    base_preprocessing_layer.keras_kpl_gauge.get_cell("Discretization").set(
        True)

    # Check dtype only after base layer parses it; dtype parsing is complex.
    if output_mode == INT and not tf.as_dtype(self.compute_dtype).is_integer:
      input_dtype = kwargs["dtype"]
      raise ValueError("When `output_mode='int'`, `dtype` should be an integer "
                       f"type. Received: dtype={input_dtype}")

    # 'output_mode' must be one of (INT, ONE_HOT, MULTI_HOT, COUNT)
    layer_utils.validate_string_arg(
        output_mode,
        allowable_strings=(INT, ONE_HOT, MULTI_HOT, COUNT),
        layer_name=self.__class__.__name__,
        arg_name="output_mode")

    if sparse and output_mode == INT:
      raise ValueError(f"`sparse` may only be true if `output_mode` is "
                       f"`'one_hot'`, `'multi_hot'`, or `'count'`. "
                       f"Received: sparse={sparse} and "
                       f"output_mode={output_mode}")

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
    self.output_mode = output_mode
    self.sparse = sparse

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

  # We override this method solely to generate a docstring.
  def adapt(self, data, batch_size=None, steps=None):
    """Computes bin boundaries from quantiles in a input dataset.

    Calling `adapt()` on a `Discretization` layer is an alternative to passing
    in a `bin_boundaries` argument during construction. A `Discretization` layer
    should always be either adapted over a dataset or passed `bin_boundaries`.

    During `adapt()`, the layer will estimate the quantile boundaries of the
    input dataset. The number of quantiles can be controlled via the `num_bins`
    argument, and the error tolerance for quantile boundaries can be controlled
    via the `epsilon` argument.

    In order to make `Discretization` efficient in any distribution context, the
    computed boundaries are kept static with respect to any compiled `tf.Graph`s
    that call the layer. As a consequence, if the layer is adapted a second
    time, any models using the layer should be re-compiled. For more information
    see `tf.keras.layers.experimental.preprocessing.PreprocessingLayer.adapt`.

    `adapt()` is meant only as a single machine utility to compute layer state.
    To analyze a dataset that cannot fit on a single machine, see
    [Tensorflow Transform](https://www.tensorflow.org/tfx/transform/get_started)
    for a multi-machine, map-reduce solution.

    Arguments:
      data: The data to train on. It can be passed either as a
          `tf.data.Dataset`, or as a numpy array.
      batch_size: Integer or `None`.
          Number of samples per state update.
          If unspecified, `batch_size` will default to 32.
          Do not specify the `batch_size` if your data is in the
          form of datasets, generators, or `keras.utils.Sequence` instances
          (since they generate batches).
      steps: Integer or `None`.
          Total number of steps (batches of samples)
          When training with input tensors such as
          TensorFlow data tensors, the default `None` is equal to
          the number of samples in your dataset divided by
          the batch size, or 1 if that cannot be determined. If x is a
          `tf.data` dataset, and 'steps' is None, the epoch will run until
          the input dataset is exhausted. When passing an infinitely
          repeating dataset, you must specify the `steps` argument. This
          argument is not supported with array inputs.
    """
    super().adapt(data, batch_size=batch_size, steps=steps)

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
        "output_mode": self.output_mode,
        "sparse": self.sparse,
    })
    return config

  def compute_output_shape(self, input_shape):
    return input_shape

  def compute_output_signature(self, input_spec):
    output_shape = self.compute_output_shape(input_spec.shape.as_list())
    if isinstance(input_spec, tf.SparseTensorSpec):
      return tf.SparseTensorSpec(
          shape=output_shape, dtype=self.compute_dtype)
    return tf.TensorSpec(shape=output_shape, dtype=self.compute_dtype)

  def call(self, inputs):
    def bucketize(inputs):
      return tf.raw_ops.Bucketize(input=inputs, boundaries=self.bin_boundaries)

    if tf_utils.is_ragged(inputs):
      indices = tf.ragged.map_flat_values(bucketize, inputs)
    elif tf_utils.is_sparse(inputs):
      indices = tf.SparseTensor(
          indices=tf.identity(inputs.indices),
          values=bucketize(inputs.values),
          dense_shape=tf.identity(inputs.dense_shape))
    else:
      indices = bucketize(inputs)

    return utils.encode_categorical_inputs(
        indices,
        output_mode=self.output_mode,
        depth=len(self.bin_boundaries) + 1,
        sparse=self.sparse,
        dtype=self.compute_dtype)
