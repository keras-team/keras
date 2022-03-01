# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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
"""Wrapper layer to apply every temporal slice of an input."""
# pylint: disable=g-classes-have-attributes,g-direct-tensorflow-import

from keras import backend
from keras.engine.base_layer import Layer
from keras.engine.input_spec import InputSpec
from keras.layers.rnn.base_wrapper import Wrapper
from keras.utils import generic_utils
from keras.utils import layer_utils
from keras.utils import tf_utils
import tensorflow.compat.v2 as tf

from tensorflow.python.util.tf_export import keras_export


@keras_export('keras.layers.TimeDistributed')
class TimeDistributed(Wrapper):
  """This wrapper allows to apply a layer to every temporal slice of an input.

  Every input should be at least 3D, and the dimension of index one of the
  first input will be considered to be the temporal dimension.

  Consider a batch of 32 video samples, where each sample is a 128x128 RGB image
  with `channels_last` data format, across 10 timesteps.
  The batch input shape is `(32, 10, 128, 128, 3)`.

  You can then use `TimeDistributed` to apply the same `Conv2D` layer to each
  of the 10 timesteps, independently:

  >>> inputs = tf.keras.Input(shape=(10, 128, 128, 3))
  >>> conv_2d_layer = tf.keras.layers.Conv2D(64, (3, 3))
  >>> outputs = tf.keras.layers.TimeDistributed(conv_2d_layer)(inputs)
  >>> outputs.shape
  TensorShape([None, 10, 126, 126, 64])

  Because `TimeDistributed` applies the same instance of `Conv2D` to each of the
  timestamps, the same set of weights are used at each timestamp.

  Args:
    layer: a `tf.keras.layers.Layer` instance.

  Call arguments:
    inputs: Input tensor of shape (batch, time, ...) or nested tensors,
      and each of which has shape (batch, time, ...).
    training: Python boolean indicating whether the layer should behave in
      training mode or in inference mode. This argument is passed to the
      wrapped layer (only if the layer supports this argument).
    mask: Binary tensor of shape `(samples, timesteps)` indicating whether
      a given timestep should be masked. This argument is passed to the
      wrapped layer (only if the layer supports this argument).

  Raises:
    ValueError: If not initialized with a `tf.keras.layers.Layer` instance.
  """

  def __init__(self, layer, **kwargs):
    if not isinstance(layer, Layer):
      raise ValueError(
          'Please initialize `TimeDistributed` layer with a '
          f'`tf.keras.layers.Layer` instance. Received: {layer}')
    super(TimeDistributed, self).__init__(layer, **kwargs)
    self.supports_masking = True

    # It is safe to use the fast, reshape-based approach with all of our
    # built-in Layers.
    self._always_use_reshape = (
        layer_utils.is_builtin_layer(layer) and
        not getattr(layer, 'stateful', False))

  def _get_shape_tuple(self, init_tuple, tensor, start_idx, int_shape=None):
    """Finds non-specific dimensions in the static shapes.

    The static shapes are replaced with the corresponding dynamic shapes of the
    tensor.
    Args:
      init_tuple: a tuple, the first part of the output shape
      tensor: the tensor from which to get the (static and dynamic) shapes
        as the last part of the output shape
      start_idx: int, which indicate the first dimension to take from
        the static shape of the tensor
      int_shape: an alternative static shape to take as the last part
        of the output shape
    Returns:
      The new int_shape with the first part from init_tuple
      and the last part from either `int_shape` (if provided)
      or `tensor.shape`, where every `None` is replaced by
      the corresponding dimension from `tf.shape(tensor)`.
    """
    # replace all None in int_shape by backend.shape
    if int_shape is None:
      int_shape = backend.int_shape(tensor)[start_idx:]
    if isinstance(int_shape, tf.TensorShape):
      int_shape = int_shape.as_list()
    if not any(not s for s in int_shape):
      return init_tuple + tuple(int_shape)
    shape = backend.shape(tensor)
    int_shape = list(int_shape)
    for i, s in enumerate(int_shape):
      if not s:
        int_shape[i] = shape[start_idx + i]
    return init_tuple + tuple(int_shape)

  def _remove_timesteps(self, dims):
    dims = dims.as_list()
    return tf.TensorShape([dims[0]] + dims[2:])

  def build(self, input_shape):
    input_shape = tf_utils.convert_shapes(input_shape, to_tuples=False)
    input_dims = tf.nest.flatten(
        tf.nest.map_structure(lambda x: x.ndims, input_shape))
    if any(dim < 3 for dim in input_dims):
      raise ValueError(
          '`TimeDistributed` Layer should be passed an `input_shape ` '
          f'with at least 3 dimensions, received: {input_shape}')
    # Don't enforce the batch or time dimension.
    self.input_spec = tf.nest.map_structure(
        lambda x: InputSpec(shape=[None, None] + x.as_list()[2:]), input_shape)
    child_input_shape = tf.nest.map_structure(self._remove_timesteps,
                                              input_shape)
    child_input_shape = tf_utils.convert_shapes(child_input_shape)
    super(TimeDistributed, self).build(tuple(child_input_shape))
    self.built = True

  def compute_output_shape(self, input_shape):
    input_shape = tf_utils.convert_shapes(input_shape, to_tuples=False)

    child_input_shape = tf.nest.map_structure(self._remove_timesteps,
                                              input_shape)
    child_output_shape = self.layer.compute_output_shape(child_input_shape)
    child_output_shape = tf_utils.convert_shapes(
        child_output_shape, to_tuples=False)
    timesteps = tf_utils.convert_shapes(input_shape)
    timesteps = tf.nest.flatten(timesteps)[1]

    def insert_timesteps(dims):
      dims = dims.as_list()
      return tf.TensorShape([dims[0], timesteps] + dims[1:])

    return tf.nest.map_structure(insert_timesteps, child_output_shape)

  def call(self, inputs, training=None, mask=None):
    kwargs = {}
    if generic_utils.has_arg(self.layer.call, 'training'):
      kwargs['training'] = training

    input_shape = tf.nest.map_structure(
        lambda x: tf.TensorShape(backend.int_shape(x)), inputs)
    batch_size = tf_utils.convert_shapes(input_shape)
    batch_size = tf.nest.flatten(batch_size)[0]
    if batch_size and not self._always_use_reshape:
      inputs, row_lengths = backend.convert_inputs_if_ragged(inputs)
      is_ragged_input = row_lengths is not None
      input_length = tf_utils.convert_shapes(input_shape)
      input_length = tf.nest.flatten(input_length)[1]

      # batch size matters, use rnn-based implementation
      def step(x, _):
        output = self.layer(x, **kwargs)
        return output, []

      _, outputs, _ = backend.rnn(
          step,
          inputs,
          initial_states=[],
          input_length=row_lengths[0] if is_ragged_input else input_length,
          mask=mask,
          unroll=False)
      # pylint: disable=g-long-lambda
      y = tf.nest.map_structure(
          lambda output: backend.maybe_convert_to_ragged(
              is_ragged_input, output, row_lengths), outputs)
    else:
      # No batch size specified, therefore the layer will be able
      # to process batches of any size.
      # We can go with reshape-based implementation for performance.
      is_ragged_input = tf.nest.map_structure(
          lambda x: isinstance(x, tf.RaggedTensor), inputs)
      is_ragged_input = tf.nest.flatten(is_ragged_input)
      if all(is_ragged_input):
        input_values = tf.nest.map_structure(lambda x: x.values, inputs)
        input_row_lenghts = tf.nest.map_structure(
            lambda x: x.nested_row_lengths()[0], inputs)
        y = self.layer(input_values, **kwargs)
        y = tf.nest.map_structure(tf.RaggedTensor.from_row_lengths, y,
                                  input_row_lenghts)
      elif any(is_ragged_input):
        raise ValueError('All inputs has to be either ragged or not, '
                         f'but not mixed. Received: {inputs}')
      else:
        input_length = tf_utils.convert_shapes(input_shape)
        input_length = tf.nest.flatten(input_length)[1]
        if not input_length:
          input_length = tf.nest.map_structure(lambda x: tf.shape(x)[1], inputs)
          input_length = generic_utils.to_list(tf.nest.flatten(input_length))[0]

        inner_input_shape = tf.nest.map_structure(
            lambda x: self._get_shape_tuple((-1,), x, 2), inputs)
        # Shape: (num_samples * timesteps, ...). And track the
        # transformation in self._input_map.
        inputs = tf.__internal__.nest.map_structure_up_to(
            inputs, tf.reshape, inputs, inner_input_shape)
        # (num_samples * timesteps, ...)
        if generic_utils.has_arg(self.layer.call, 'mask') and mask is not None:
          inner_mask_shape = self._get_shape_tuple((-1,), mask, 2)
          kwargs['mask'] = backend.reshape(mask, inner_mask_shape)

        y = self.layer(inputs, **kwargs)

        # Shape: (num_samples, timesteps, ...)
        output_shape = self.compute_output_shape(input_shape)
        # pylint: disable=g-long-lambda
        output_shape = tf.nest.map_structure(
            lambda tensor, int_shape: self._get_shape_tuple(
                (-1, input_length), tensor, 1, int_shape[2:]), y, output_shape)
        y = tf.__internal__.nest.map_structure_up_to(y, tf.reshape, y,
                                                     output_shape)
        if not tf.executing_eagerly():
          # Set the static shape for the result since it might be lost during
          # array_ops reshape, eg, some `None` dim in the result could be
          # inferred.
          tf.__internal__.nest.map_structure_up_to(
              y, lambda tensor, shape: tensor.set_shape(shape), y,
              self.compute_output_shape(input_shape))

    return y

  def compute_mask(self, inputs, mask=None):
    """Computes an output mask tensor for Embedding layer.

    This is based on the inputs, mask, and the inner layer.
    If batch size is specified:
    Simply return the input `mask`. (An rnn-based implementation with
    more than one rnn inputs is required but not supported in tf.keras yet.)
    Otherwise we call `compute_mask` of the inner layer at each time step.
    If the output mask at each time step is not `None`:
    (E.g., inner layer is Masking or RNN)
    Concatenate all of them and return the concatenation.
    If the output mask at each time step is `None` and the input mask is not
    `None`:(E.g., inner layer is Dense)
    Reduce the input_mask to 2 dimensions and return it.
    Otherwise (both the output mask and the input mask are `None`):
    (E.g., `mask` is not used at all)
    Return `None`.

    Args:
      inputs: Tensor with shape [batch size, timesteps, ...] indicating the
        input to TimeDistributed. If static shape information is available for
        "batch size", `mask` is returned unmodified.
      mask: Either None (indicating no masking) or a Tensor indicating the
        input mask for TimeDistributed. The shape can be static or dynamic.

    Returns:
      Either None (no masking), or a [batch size, timesteps, ...] Tensor with
      an output mask for the TimeDistributed layer with the shape beyond the
      second dimension being the value of the input mask shape(if the computed
      output mask is none), an output mask with the shape beyond the first
      dimension being the value of the mask shape(if mask is not None) or
      output mask with the shape beyond the first dimension being the
      value of the computed output shape.

    """
    # cases need to call the layer.compute_mask when input_mask is None:
    # Masking layer and Embedding layer with mask_zero
    input_shape = tf.nest.map_structure(
        lambda x: tf.TensorShape(backend.int_shape(x)), inputs)
    input_shape = tf_utils.convert_shapes(input_shape, to_tuples=False)
    batch_size = tf_utils.convert_shapes(input_shape)
    batch_size = tf.nest.flatten(batch_size)[0]
    is_ragged_input = tf.nest.map_structure(
        lambda x: isinstance(x, tf.RaggedTensor), inputs)
    is_ragged_input = generic_utils.to_list(tf.nest.flatten(is_ragged_input))
    if batch_size and not self._always_use_reshape or any(is_ragged_input):
      # batch size matters, we currently do not handle mask explicitly, or if
      # the layer always uses reshape approach, or the input is a ragged tensor.
      return mask
    inner_mask = mask
    if inner_mask is not None:
      inner_mask_shape = self._get_shape_tuple((-1,), mask, 2)
      inner_mask = backend.reshape(inner_mask, inner_mask_shape)
    inner_input_shape = tf.nest.map_structure(
        lambda tensor: self._get_shape_tuple((-1,), tensor, 2), inputs)
    inner_inputs = tf.__internal__.nest.map_structure_up_to(
        inputs, tf.reshape, inputs, inner_input_shape)
    output_mask = self.layer.compute_mask(inner_inputs, inner_mask)
    if output_mask is None:
      if mask is None:
        return None
      # input_mask is not None, and output_mask is None:
      # we should return a not-None mask
      output_mask = mask
      for _ in range(2, len(backend.int_shape(mask))):
        output_mask = backend.any(output_mask, axis=-1)
    else:
      # output_mask is not None. We need to reshape it
      input_length = tf_utils.convert_shapes(input_shape)
      input_length = tf.nest.flatten(input_length)[1]
      if not input_length:
        input_length = tf.nest.map_structure(lambda x: backend.shape(x)[1],
                                             inputs)
        input_length = tf.nest.flatten(input_length)[0]
      output_mask_int_shape = backend.int_shape(output_mask)
      if output_mask_int_shape is None:
        # if the output_mask does not have a static shape,
        # its shape must be the same as mask's
        if mask is not None:
          output_mask_int_shape = backend.int_shape(mask)
        else:
          input_shape = generic_utils.to_list(tf.nest.flatten(input_shape))[0]
          output_mask_int_shape = backend.compute_output_shape(input_shape)[:-1]
      output_mask_shape = self._get_shape_tuple(
          (-1, input_length), output_mask, 1, output_mask_int_shape[1:])
      output_mask = backend.reshape(output_mask, output_mask_shape)
    return output_mask
