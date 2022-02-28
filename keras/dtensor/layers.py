# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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
"""DTensor specific Keras layers."""

import functools

from keras import backend
from keras import layers
from keras.dtensor import initializers
from keras.engine import base_layer
from keras.engine import base_layer_utils
from keras.engine import input_spec
from keras.layers.convolutional import base_conv
from keras.utils import tf_inspect

import tensorflow.compat.v2 as tf

from tensorflow.dtensor import python as dtensor  # pylint: disable=g-direct-tensorflow-import


# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring
# pylint: disable=unused-argument
def make_variable(name,
                  shape=None,
                  dtype=tf.float32,
                  initializer=None,
                  layout=None,
                  trainable=None,
                  caching_device=None,
                  validate_shape=True,
                  constraint=None,
                  use_resource=None,
                  collections=None,
                  synchronization=tf.VariableSynchronization.AUTO,
                  aggregation=tf.VariableAggregation.NONE,
                  partitioner=None):
  # Note that this function is copied from keras.engine.base_layer_utils.
  # The only part that is changed are the usage of tf.Variable. The original
  # version was using tf.compat.v1.Variable for backward compat for estimator.
  initializing_from_value = False
  if initializer is not None and not callable(initializer):
    initializing_from_value = True

  if initializing_from_value:
    init_val = initializer
    variable_dtype = None
  else:
    # Instantiate initializer if provided initializer is a type object.
    if tf_inspect.isclass(initializer):
      initializer = initializer()
    init_val = functools.partial(initializer, shape, dtype=dtype, layout=layout)
    variable_dtype = dtype.base_dtype

  variable_shape = tf.TensorShape(shape)

  return dtensor.DVariable(
      initial_value=init_val,
      name=name,
      trainable=trainable,
      caching_device=caching_device,
      dtype=variable_dtype,
      validate_shape=validate_shape,
      constraint=constraint,
      synchronization=synchronization,
      aggregation=aggregation,
      shape=variable_shape if variable_shape else None)


class Layer(base_layer.Layer):

  def add_weight(self, name=None, shape=None, dtype=None, initializer=None,
                 regularizer=None, trainable=None, constraint=None,
                 use_resource=None, synchronization=None, aggregation=None,
                 layout=None, **kwargs):
    if layout:
      getter = kwargs.pop('getter', make_variable)
      # Make sure to inject the layout to the make_variable function, otherwise
      # we will have to patch the self._add_variable_with_custom_getter
      getter = functools.partial(getter, layout=layout)
    else:
      getter = base_layer_utils.make_variable
    return super().add_weight(
        name=name, shape=shape, dtype=dtype, initializer=initializer,
        regularizer=regularizer, trainable=trainable, constraint=constraint,
        use_resource=use_resource, synchronization=synchronization,
        aggregation=aggregation, getter=getter, **kwargs)


##################### Core layers ############################
class Dense(Layer, layers.Dense):

  # pylint:disable=super-init-not-called
  def __init__(self,
               units,
               activation=None,
               use_bias=True,
               kernel_initializer='glorot_uniform',
               bias_initializer='zeros',
               kernel_regularizer=None,
               bias_regularizer=None,
               activity_regularizer=None,
               kernel_constraint=None,
               bias_constraint=None,
               kernel_layout=None,
               bias_layout=None,
               **kwargs):
    kernel_initializer = initializers.get(kernel_initializer)
    bias_initializer = initializers.get(bias_initializer)
    # Note that the Layer is the first parent class here, since we would like
    # all its methods to take priority than the methods from Dense
    # (eg, `add_weight()`). We skip the Layer.__init__ here since it doesn't
    # contain any __init__ logic, and it will also cause issue for the init
    # ordering here (Layer -> base_layer.Layer -> dense.Dense). We explicitly
    # all the dense.Dense.__init__ and it will have the correct orders of init
    # for all the base classes.
    layers.Dense.__init__(
        self,
        units=units,
        activation=activation,
        use_bias=use_bias,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        kernel_regularizer=kernel_regularizer,
        bias_regularizer=bias_regularizer,
        activity_regularizer=activity_regularizer,
        kernel_constraint=kernel_constraint,
        bias_constraint=bias_constraint,
        **kwargs)
    self.kernel_layout = kernel_layout
    self.bias_layout = bias_layout

  def build(self, input_shape):
    dtype = tf.as_dtype(self.dtype or backend.floatx())
    if not (dtype.is_floating or dtype.is_complex):
      raise TypeError('A Dense layer can only be built with a floating-point '
                      f'dtype. Received: dtype={dtype}')

    input_shape = tf.TensorShape(input_shape)
    last_dim = tf.compat.dimension_value(input_shape[-1])
    if last_dim is None:
      raise ValueError('The last dimension of the inputs to a Dense layer '
                       'should be defined. Found None. '
                       f'Full input shape received: {input_shape}')
    self.input_spec = input_spec.InputSpec(min_ndim=2, axes={-1: last_dim})
    self.kernel = self.add_weight(
        'kernel',
        shape=[last_dim, self.units],
        initializer=self.kernel_initializer,
        regularizer=self.kernel_regularizer,
        constraint=self.kernel_constraint,
        layout=self.kernel_layout,
        dtype=self.dtype,
        trainable=True)
    if self.use_bias:
      self.bias = self.add_weight(
          'bias',
          shape=[self.units,],
          initializer=self.bias_initializer,
          regularizer=self.bias_regularizer,
          constraint=self.bias_constraint,
          layout=self.bias_layout,
          dtype=self.dtype,
          trainable=True)
    else:
      self.bias = None
    self.built = True


##################### Conv layers ############################
class Conv(Layer, base_conv.Conv):

  # pylint:disable=super-init-not-called
  # See Dense layer __init__ for why do this.
  def __init__(self,
               rank,
               filters,
               kernel_size,
               strides=1,
               padding='valid',
               data_format=None,
               dilation_rate=1,
               groups=1,
               activation=None,
               use_bias=True,
               kernel_initializer='glorot_uniform',
               bias_initializer='zeros',
               kernel_regularizer=None,
               bias_regularizer=None,
               activity_regularizer=None,
               kernel_constraint=None,
               bias_constraint=None,
               kernel_layout=None,
               bias_layout=None,
               trainable=True,
               name=None,
               conv_op=None,
               **kwargs):
    kernel_initializer = initializers.get(kernel_initializer)
    bias_initializer = initializers.get(bias_initializer)

    base_conv.Conv.__init__(
        self,
        rank=rank,
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        data_format=data_format,
        dilation_rate=dilation_rate,
        groups=groups,
        activation=activation,
        use_bias=use_bias,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        kernel_regularizer=kernel_regularizer,
        bias_regularizer=bias_regularizer,
        activity_regularizer=activity_regularizer,
        kernel_constraint=kernel_constraint,
        bias_constraint=bias_constraint,
        trainable=trainable,
        name=name,
        conv_op=conv_op,
        **kwargs)
    self.kernel_layout = kernel_layout
    self.bias_layout = bias_layout

  def build(self, input_shape):
    input_shape = tf.TensorShape(input_shape)
    input_channel = self._get_input_channel(input_shape)
    if input_channel % self.groups != 0:
      raise ValueError(
          'The number of input channels must be evenly divisible by the number '
          'of groups. Received groups={}, but the input has {} channels '
          '(full input shape is {}).'.format(self.groups, input_channel,
                                             input_shape))
    kernel_shape = self.kernel_size + (input_channel // self.groups,
                                       self.filters)

    # compute_output_shape contains some validation logic for the input shape,
    # and make sure the output shape has all positive dimensions.
    self.compute_output_shape(input_shape)

    self.kernel = self.add_weight(
        name='kernel',
        shape=kernel_shape,
        initializer=self.kernel_initializer,
        regularizer=self.kernel_regularizer,
        constraint=self.kernel_constraint,
        layout=self.kernel_layout,
        trainable=True,
        dtype=self.dtype)
    if self.use_bias:
      self.bias = self.add_weight(
          name='bias',
          shape=(self.filters,),
          initializer=self.bias_initializer,
          regularizer=self.bias_regularizer,
          constraint=self.bias_constraint,
          layout=self.bias_layout,
          trainable=True,
          dtype=self.dtype)
    else:
      self.bias = None
    channel_axis = self._get_channel_axis()
    self.input_spec = input_spec.InputSpec(min_ndim=self.rank + 2,
                                           axes={channel_axis: input_channel})
    self.built = True


class Conv2D(Conv):

  def __init__(self,
               filters,
               kernel_size,
               strides=(1, 1),
               padding='valid',
               data_format=None,
               dilation_rate=(1, 1),
               groups=1,
               activation=None,
               use_bias=True,
               kernel_initializer='glorot_uniform',
               bias_initializer='zeros',
               kernel_regularizer=None,
               bias_regularizer=None,
               activity_regularizer=None,
               kernel_constraint=None,
               bias_constraint=None,
               kernel_layout=None,
               bias_layout=None,
               **kwargs):
    super(Conv2D, self).__init__(
        rank=2,
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        data_format=data_format,
        dilation_rate=dilation_rate,
        groups=groups,
        activation=activation,
        use_bias=use_bias,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        kernel_regularizer=kernel_regularizer,
        bias_regularizer=bias_regularizer,
        activity_regularizer=activity_regularizer,
        kernel_constraint=kernel_constraint,
        bias_constraint=bias_constraint,
        kernel_layout=kernel_layout,
        bias_layout=bias_layout,
        **kwargs)


##################### Aliens ############################
# So far we only add those layers need by mnist example.
# Might populate more later. In future, we might consider patch the tf.keras API
# within the dtensor_api.init() function, so that end user will only access
# via the TF public API.
Dropout = layers.Dropout
Flatten = layers.Flatten
MaxPooling2D = layers.MaxPooling2D
