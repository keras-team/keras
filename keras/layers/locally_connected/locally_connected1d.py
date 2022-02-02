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
# pylint: disable=g-classes-have-attributes,g-direct-tensorflow-import
"""Locally-connected layer for 1D input."""

from keras import activations
from keras import backend
from keras import constraints
from keras import initializers
from keras import regularizers
from keras.engine.base_layer import Layer
from keras.engine.input_spec import InputSpec
from keras.layers.locally_connected import locally_connected_utils
from keras.utils import conv_utils
from keras.utils import tf_utils

from tensorflow.python.util.tf_export import keras_export


@keras_export('keras.layers.LocallyConnected1D')
class LocallyConnected1D(Layer):
  """Locally-connected layer for 1D inputs.

  The `LocallyConnected1D` layer works similarly to
  the `Conv1D` layer, except that weights are unshared,
  that is, a different set of filters is applied at each different patch
  of the input.

  Note: layer attributes cannot be modified after the layer has been called
  once (except the `trainable` attribute).

  Example:
  ```python
      # apply a unshared weight convolution 1d of length 3 to a sequence with
      # 10 timesteps, with 64 output filters
      model = Sequential()
      model.add(LocallyConnected1D(64, 3, input_shape=(10, 32)))
      # now model.output_shape == (None, 8, 64)
      # add a new conv1d on top
      model.add(LocallyConnected1D(32, 3))
      # now model.output_shape == (None, 6, 32)
  ```

  Args:
      filters: Integer, the dimensionality of the output space (i.e. the number
        of output filters in the convolution).
      kernel_size: An integer or tuple/list of a single integer, specifying the
        length of the 1D convolution window.
      strides: An integer or tuple/list of a single integer, specifying the
        stride length of the convolution.
      padding: Currently only supports `"valid"` (case-insensitive). `"same"`
        may be supported in the future. `"valid"` means no padding.
      data_format: A string, one of `channels_last` (default) or
        `channels_first`. The ordering of the dimensions in the inputs.
        `channels_last` corresponds to inputs with shape `(batch, length,
        channels)` while `channels_first` corresponds to inputs with shape
        `(batch, channels, length)`. It defaults to the `image_data_format`
        value found in your Keras config file at `~/.keras/keras.json`. If you
        never set it, then it will be "channels_last".
      activation: Activation function to use. If you don't specify anything, no
        activation is applied
          (ie. "linear" activation: `a(x) = x`).
      use_bias: Boolean, whether the layer uses a bias vector.
      kernel_initializer: Initializer for the `kernel` weights matrix.
      bias_initializer: Initializer for the bias vector.
      kernel_regularizer: Regularizer function applied to the `kernel` weights
        matrix.
      bias_regularizer: Regularizer function applied to the bias vector.
      activity_regularizer: Regularizer function applied to the output of the
        layer (its "activation")..
      kernel_constraint: Constraint function applied to the kernel matrix.
      bias_constraint: Constraint function applied to the bias vector.
      implementation: implementation mode, either `1`, `2`, or `3`. `1` loops
        over input spatial locations to perform the forward pass. It is
        memory-efficient but performs a lot of (small) ops.  `2` stores layer
        weights in a dense but sparsely-populated 2D matrix and implements the
        forward pass as a single matrix-multiply. It uses a lot of RAM but
        performs few (large) ops.  `3` stores layer weights in a sparse tensor
        and implements the forward pass as a single sparse matrix-multiply.
          How to choose:
          `1`: large, dense models,
          `2`: small models,
          `3`: large, sparse models,  where "large" stands for large
            input/output activations (i.e. many `filters`, `input_filters`,
            large `input_size`, `output_size`), and "sparse" stands for few
            connections between inputs and outputs, i.e. small ratio `filters *
            input_filters * kernel_size / (input_size * strides)`, where inputs
            to and outputs of the layer are assumed to have shapes `(input_size,
            input_filters)`, `(output_size, filters)` respectively.  It is
            recommended to benchmark each in the setting of interest to pick the
            most efficient one (in terms of speed and memory usage). Correct
            choice of implementation can lead to dramatic speed improvements
            (e.g. 50X), potentially at the expense of RAM.  Also, only
            `padding="valid"` is supported by `implementation=1`.
  Input shape:
      3D tensor with shape: `(batch_size, steps, input_dim)`
  Output shape:
      3D tensor with shape: `(batch_size, new_steps, filters)` `steps` value
        might have changed due to padding or strides.
  """

  def __init__(self,
               filters,
               kernel_size,
               strides=1,
               padding='valid',
               data_format=None,
               activation=None,
               use_bias=True,
               kernel_initializer='glorot_uniform',
               bias_initializer='zeros',
               kernel_regularizer=None,
               bias_regularizer=None,
               activity_regularizer=None,
               kernel_constraint=None,
               bias_constraint=None,
               implementation=1,
               **kwargs):
    super(LocallyConnected1D, self).__init__(**kwargs)
    self.filters = filters
    self.kernel_size = conv_utils.normalize_tuple(kernel_size, 1, 'kernel_size')
    self.strides = conv_utils.normalize_tuple(
        strides, 1, 'strides', allow_zero=True)
    self.padding = conv_utils.normalize_padding(padding)
    if self.padding != 'valid' and implementation == 1:
      raise ValueError('Invalid border mode for LocallyConnected1D '
                       '(only "valid" is supported if implementation is 1): ' +
                       padding)
    self.data_format = conv_utils.normalize_data_format(data_format)
    self.activation = activations.get(activation)
    self.use_bias = use_bias
    self.kernel_initializer = initializers.get(kernel_initializer)
    self.bias_initializer = initializers.get(bias_initializer)
    self.kernel_regularizer = regularizers.get(kernel_regularizer)
    self.bias_regularizer = regularizers.get(bias_regularizer)
    self.activity_regularizer = regularizers.get(activity_regularizer)
    self.kernel_constraint = constraints.get(kernel_constraint)
    self.bias_constraint = constraints.get(bias_constraint)
    self.implementation = implementation
    self.input_spec = InputSpec(ndim=3)

  @property
  def _use_input_spec_as_call_signature(self):
    return False

  @tf_utils.shape_type_conversion
  def build(self, input_shape):
    if self.data_format == 'channels_first':
      input_dim, input_length = input_shape[1], input_shape[2]
    else:
      input_dim, input_length = input_shape[2], input_shape[1]

    if input_dim is None:
      raise ValueError(
          'Axis 2 of input should be fully-defined. '
          'Found shape:', input_shape)
    self.output_length = conv_utils.conv_output_length(input_length,
                                                       self.kernel_size[0],
                                                       self.padding,
                                                       self.strides[0])

    if self.output_length <= 0:
      raise ValueError(
          f'One of the dimensions in the output is <= 0 '
          f'due to downsampling in {self.name}. Consider '
          f'increasing the input size. '
          f'Received input shape {input_shape} which would produce '
          f'output shape with a zero or negative value in a '
          f'dimension.')

    if self.implementation == 1:
      self.kernel_shape = (self.output_length, self.kernel_size[0] * input_dim,
                           self.filters)

      self.kernel = self.add_weight(
          shape=self.kernel_shape,
          initializer=self.kernel_initializer,
          name='kernel',
          regularizer=self.kernel_regularizer,
          constraint=self.kernel_constraint)

    elif self.implementation == 2:
      if self.data_format == 'channels_first':
        self.kernel_shape = (input_dim, input_length, self.filters,
                             self.output_length)
      else:
        self.kernel_shape = (input_length, input_dim, self.output_length,
                             self.filters)

      self.kernel = self.add_weight(
          shape=self.kernel_shape,
          initializer=self.kernel_initializer,
          name='kernel',
          regularizer=self.kernel_regularizer,
          constraint=self.kernel_constraint)

      self.kernel_mask = locally_connected_utils.get_locallyconnected_mask(
          input_shape=(input_length,),
          kernel_shape=self.kernel_size,
          strides=self.strides,
          padding=self.padding,
          data_format=self.data_format,
      )

    elif self.implementation == 3:
      self.kernel_shape = (self.output_length * self.filters,
                           input_length * input_dim)

      self.kernel_idxs = sorted(
          conv_utils.conv_kernel_idxs(
              input_shape=(input_length,),
              kernel_shape=self.kernel_size,
              strides=self.strides,
              padding=self.padding,
              filters_in=input_dim,
              filters_out=self.filters,
              data_format=self.data_format))

      self.kernel = self.add_weight(
          shape=(len(self.kernel_idxs),),
          initializer=self.kernel_initializer,
          name='kernel',
          regularizer=self.kernel_regularizer,
          constraint=self.kernel_constraint)

    else:
      raise ValueError('Unrecognized implementation mode: %d.' %
                       self.implementation)

    if self.use_bias:
      self.bias = self.add_weight(
          shape=(self.output_length, self.filters),
          initializer=self.bias_initializer,
          name='bias',
          regularizer=self.bias_regularizer,
          constraint=self.bias_constraint)
    else:
      self.bias = None

    if self.data_format == 'channels_first':
      self.input_spec = InputSpec(ndim=3, axes={1: input_dim})
    else:
      self.input_spec = InputSpec(ndim=3, axes={-1: input_dim})
    self.built = True

  @tf_utils.shape_type_conversion
  def compute_output_shape(self, input_shape):
    if self.data_format == 'channels_first':
      input_length = input_shape[2]
    else:
      input_length = input_shape[1]

    length = conv_utils.conv_output_length(input_length, self.kernel_size[0],
                                           self.padding, self.strides[0])

    if self.data_format == 'channels_first':
      return (input_shape[0], self.filters, length)
    elif self.data_format == 'channels_last':
      return (input_shape[0], length, self.filters)

  def call(self, inputs):
    if self.implementation == 1:
      output = backend.local_conv(
          inputs, self.kernel, self.kernel_size, self.strides,
          (self.output_length,), self.data_format)

    elif self.implementation == 2:
      output = locally_connected_utils.local_conv_matmul(
          inputs, self.kernel, self.kernel_mask,
          self.compute_output_shape(inputs.shape))

    elif self.implementation == 3:
      output = locally_connected_utils.local_conv_sparse_matmul(
          inputs, self.kernel, self.kernel_idxs, self.kernel_shape,
          self.compute_output_shape(inputs.shape))

    else:
      raise ValueError('Unrecognized implementation mode: %d.' %
                       self.implementation)

    if self.use_bias:
      output = backend.bias_add(output, self.bias, data_format=self.data_format)

    output = self.activation(output)
    return output

  def get_config(self):
    config = {
        'filters':
            self.filters,
        'kernel_size':
            self.kernel_size,
        'strides':
            self.strides,
        'padding':
            self.padding,
        'data_format':
            self.data_format,
        'activation':
            activations.serialize(self.activation),
        'use_bias':
            self.use_bias,
        'kernel_initializer':
            initializers.serialize(self.kernel_initializer),
        'bias_initializer':
            initializers.serialize(self.bias_initializer),
        'kernel_regularizer':
            regularizers.serialize(self.kernel_regularizer),
        'bias_regularizer':
            regularizers.serialize(self.bias_regularizer),
        'activity_regularizer':
            regularizers.serialize(self.activity_regularizer),
        'kernel_constraint':
            constraints.serialize(self.kernel_constraint),
        'bias_constraint':
            constraints.serialize(self.bias_constraint),
        'implementation':
            self.implementation
    }
    base_config = super(LocallyConnected1D, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))
