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
# =============================================================================
# pylint: disable=g-classes-have-attributes
"""Contains the core layers: Dense, Dropout.

Also contains their functional aliases.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v2 as tf

import warnings

from keras import layers as keras_layers
from keras.legacy_tf_layers import base
from tensorflow.python.util.tf_export import keras_export
from tensorflow.python.util.tf_export import tf_export


@keras_export(v1=['keras.__internal__.legacy.layers.Dense'])
@tf_export(v1=['layers.Dense'])
class Dense(keras_layers.Dense, base.Layer):
  """Densely-connected layer class.

  This layer implements the operation:
  `outputs = activation(inputs * kernel + bias)`
  Where `activation` is the activation function passed as the `activation`
  argument (if not `None`), `kernel` is a weights matrix created by the layer,
  and `bias` is a bias vector created by the layer
  (only if `use_bias` is `True`).

  Args:
    units: Integer or Long, dimensionality of the output space.
    activation: Activation function (callable). Set it to None to maintain a
      linear activation.
    use_bias: Boolean, whether the layer uses a bias.
    kernel_initializer: Initializer function for the weight matrix.
      If `None` (default), weights are initialized using the default
      initializer used by `tf.compat.v1.get_variable`.
    bias_initializer: Initializer function for the bias.
    kernel_regularizer: Regularizer function for the weight matrix.
    bias_regularizer: Regularizer function for the bias.
    activity_regularizer: Regularizer function for the output.
    kernel_constraint: An optional projection function to be applied to the
        kernel after being updated by an `Optimizer` (e.g. used to implement
        norm constraints or value constraints for layer weights). The function
        must take as input the unprojected variable and must return the
        projected variable (which must have the same shape). Constraints are
        not safe to use when doing asynchronous distributed training.
    bias_constraint: An optional projection function to be applied to the
        bias after being updated by an `Optimizer`.
    trainable: Boolean, if `True` also add variables to the graph collection
      `GraphKeys.TRAINABLE_VARIABLES` (see `tf.Variable`).
    name: String, the name of the layer. Layers with the same name will
      share weights, but to avoid mistakes we require reuse=True in such cases.
    _reuse: Boolean, whether to reuse the weights of a previous layer
      by the same name.

  Properties:
    units: Python integer, dimensionality of the output space.
    activation: Activation function (callable).
    use_bias: Boolean, whether the layer uses a bias.
    kernel_initializer: Initializer instance (or name) for the kernel matrix.
    bias_initializer: Initializer instance (or name) for the bias.
    kernel_regularizer: Regularizer instance for the kernel matrix (callable)
    bias_regularizer: Regularizer instance for the bias (callable).
    activity_regularizer: Regularizer instance for the output (callable)
    kernel_constraint: Constraint function for the kernel matrix.
    bias_constraint: Constraint function for the bias.
    kernel: Weight matrix (TensorFlow variable or tensor).
    bias: Bias vector, if applicable (TensorFlow variable or tensor).


  @compatibility(TF2)
  This API is a legacy api that is only compatible with eager execution and
  `tf.function` if you combine it with
  `tf.compat.v1.keras.utils.track_tf1_style_variables`

  Please refer to [tf.layers model mapping section of the migration guide]
  (https://www.tensorflow.org/guide/migrate/model_mapping)
  to learn how to use your TensorFlow v1 model in TF2 with Keras.

  The corresponding TensorFlow v2 layer is `tf.keras.layers.Dense`.


  #### Structural Mapping to Native TF2

  None of the supported arguments have changed name.

  Before:

  ```python
   dense = tf.compat.v1.layers.Dense(units=3)
  ```

  After:

  ```python
   dense = tf.keras.layers.Dense(units=3)
  ```

  @end_compatibility
  """

  def __init__(self, units,
               activation=None,
               use_bias=True,
               kernel_initializer=None,
               bias_initializer=tf.compat.v1.zeros_initializer(),
               kernel_regularizer=None,
               bias_regularizer=None,
               activity_regularizer=None,
               kernel_constraint=None,
               bias_constraint=None,
               trainable=True,
               name=None,
               **kwargs):
    super(Dense, self).__init__(units=units,
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
                                **kwargs)


@keras_export(v1=['keras.__internal__.legacy.layers.dense'])
@tf_export(v1=['layers.dense'])
def dense(
    inputs, units,
    activation=None,
    use_bias=True,
    kernel_initializer=None,
    bias_initializer=tf.compat.v1.zeros_initializer(),
    kernel_regularizer=None,
    bias_regularizer=None,
    activity_regularizer=None,
    kernel_constraint=None,
    bias_constraint=None,
    trainable=True,
    name=None,
    reuse=None):
  """Functional interface for the densely-connected layer.

  This layer implements the operation:
  `outputs = activation(inputs * kernel + bias)`
  where `activation` is the activation function passed as the `activation`
  argument (if not `None`), `kernel` is a weights matrix created by the layer,
  and `bias` is a bias vector created by the layer
  (only if `use_bias` is `True`).

  Args:
    inputs: Tensor input.
    units: Integer or Long, dimensionality of the output space.
    activation: Activation function (callable). Set it to None to maintain a
      linear activation.
    use_bias: Boolean, whether the layer uses a bias.
    kernel_initializer: Initializer function for the weight matrix.
      If `None` (default), weights are initialized using the default
      initializer used by `tf.compat.v1.get_variable`.
    bias_initializer: Initializer function for the bias.
    kernel_regularizer: Regularizer function for the weight matrix.
    bias_regularizer: Regularizer function for the bias.
    activity_regularizer: Regularizer function for the output.
    kernel_constraint: An optional projection function to be applied to the
        kernel after being updated by an `Optimizer` (e.g. used to implement
        norm constraints or value constraints for layer weights). The function
        must take as input the unprojected variable and must return the
        projected variable (which must have the same shape). Constraints are
        not safe to use when doing asynchronous distributed training.
    bias_constraint: An optional projection function to be applied to the
        bias after being updated by an `Optimizer`.
    trainable: Boolean, if `True` also add variables to the graph collection
      `GraphKeys.TRAINABLE_VARIABLES` (see `tf.Variable`).
    name: String, the name of the layer.
    reuse: Boolean, whether to reuse the weights of a previous layer
      by the same name.

  Returns:
    Output tensor the same shape as `inputs` except the last dimension is of
    size `units`.

  Raises:
    ValueError: if eager execution is enabled.


  @compatibility(TF2)
  This API is a legacy api that is only compatible with eager execution and
  `tf.function` if you combine it with
  `tf.compat.v1.keras.utils.track_tf1_style_variables`

  Please refer to [tf.layers model mapping section of the migration guide]
  (https://www.tensorflow.org/guide/migrate/model_mapping)
  to learn how to use your TensorFlow v1 model in TF2 with Keras.

  The corresponding TensorFlow v2 layer is `tf.keras.layers.Dense`.


  #### Structural Mapping to Native TF2

  None of the supported arguments have changed name.

  Before:

  ```python
   y = tf.compat.v1.layers.dense(x, units=3)
  ```

  After:

  To migrate code using TF1 functional layers use the [Keras Functional API]
  (https://www.tensorflow.org/guide/keras/functional):

  ```python
   x = tf.keras.Input((28,))
   y = tf.keras.layers.Dense(units=3)(x)
   model = tf.keras.Model(x, y)
  ```
  @end_compatibility

  """
  warnings.warn(
      '`tf.layers.dense` is deprecated and '
      'will be removed in a future version. '
      'Please use `tf.keras.layers.Dense` instead.',
      stacklevel=2)
  layer = Dense(units,
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
                _scope=name,
                _reuse=reuse)
  return layer(inputs)


@keras_export(v1=['keras.__internal__.legacy.layers.Dropout'])
@tf_export(v1=['layers.Dropout'])
class Dropout(keras_layers.Dropout, base.Layer):
  """Applies Dropout to the input.

  Dropout consists in randomly setting a fraction `rate` of input units to 0
  at each update during training time, which helps prevent overfitting.
  The units that are kept are scaled by `1 / (1 - rate)`, so that their
  sum is unchanged at training time and inference time.

  Args:
    rate: The dropout rate, between 0 and 1. E.g. `rate=0.1` would drop out
      10% of input units.
    noise_shape: 1D tensor of type `int32` representing the shape of the
      binary dropout mask that will be multiplied with the input.
      For instance, if your inputs have shape
      `(batch_size, timesteps, features)`, and you want the dropout mask
      to be the same for all timesteps, you can use
      `noise_shape=[batch_size, 1, features]`.
    seed: A Python integer. Used to create random seeds. See
      `tf.compat.v1.set_random_seed`.
      for behavior.
    name: The name of the layer (string).


  @compatibility(TF2)
  This API is a legacy api that is only compatible with eager execution and
  `tf.function` if you combine it with
  `tf.compat.v1.keras.utils.track_tf1_style_variables`

  Please refer to [tf.layers model mapping section of the migration guide]
  (https://www.tensorflow.org/guide/migrate/model_mapping)
  to learn how to use your TensorFlow v1 model in TF2 with Keras.

  The corresponding TensorFlow v2 layer is `tf.keras.layers.Dropout`.


  #### Structural Mapping to Native TF2

  None of the supported arguments have changed name.

  Before:

  ```python
   dropout = tf.compat.v1.layers.Dropout()
  ```

  After:

  ```python
   dropout = tf.keras.layers.Dropout()
  ```
  @end_compatibility
  """

  def __init__(self, rate=0.5,
               noise_shape=None,
               seed=None,
               name=None,
               **kwargs):
    super(Dropout, self).__init__(rate=rate,
                                  noise_shape=noise_shape,
                                  seed=seed,
                                  name=name,
                                  **kwargs)

  def call(self, inputs, training=False):
    return super(Dropout, self).call(inputs, training=training)


@keras_export(v1=['keras.__internal__.legacy.layers.dropout'])
@tf_export(v1=['layers.dropout'])
def dropout(inputs,
            rate=0.5,
            noise_shape=None,
            seed=None,
            training=False,
            name=None):
  """Applies Dropout to the input.

  Dropout consists in randomly setting a fraction `rate` of input units to 0
  at each update during training time, which helps prevent overfitting.
  The units that are kept are scaled by `1 / (1 - rate)`, so that their
  sum is unchanged at training time and inference time.

  Args:
    inputs: Tensor input.
    rate: The dropout rate, between 0 and 1. E.g. "rate=0.1" would drop out
      10% of input units.
    noise_shape: 1D tensor of type `int32` representing the shape of the
      binary dropout mask that will be multiplied with the input.
      For instance, if your inputs have shape
      `(batch_size, timesteps, features)`, and you want the dropout mask
      to be the same for all timesteps, you can use
      `noise_shape=[batch_size, 1, features]`.
    seed: A Python integer. Used to create random seeds. See
      `tf.compat.v1.set_random_seed`
      for behavior.
    training: Either a Python boolean, or a TensorFlow boolean scalar tensor
      (e.g. a placeholder). Whether to return the output in training mode
      (apply dropout) or in inference mode (return the input untouched).
    name: The name of the layer (string).

  Returns:
    Output tensor.

  Raises:
    ValueError: if eager execution is enabled.

  @compatibility(TF2)
  This API is a legacy api that is only compatible with eager execution and
  `tf.function` if you combine it with
  `tf.compat.v1.keras.utils.track_tf1_style_variables`

  Please refer to [tf.layers model mapping section of the migration guide]
  (https://www.tensorflow.org/guide/migrate/model_mapping)
  to learn how to use your TensorFlow v1 model in TF2 with Keras.

  The corresponding TensorFlow v2 layer is `tf.keras.layers.Dropout`.


  #### Structural Mapping to Native TF2

  None of the supported arguments have changed name.

  Before:

  ```python
   y = tf.compat.v1.layers.dropout(x)
  ```

  After:

  To migrate code using TF1 functional layers use the [Keras Functional API]
  (https://www.tensorflow.org/guide/keras/functional):

  ```python
   x = tf.keras.Input((28, 28, 1))
   y = tf.keras.layers.Dropout()(x)
   model = tf.keras.Model(x, y)
  ```
  @end_compatibility
  """
  warnings.warn(
      '`tf.layers.dropout` is deprecated and '
      'will be removed in a future version. '
      'Please use `tf.keras.layers.Dropout` instead.',
      stacklevel=2)
  layer = Dropout(rate, noise_shape=noise_shape, seed=seed, name=name)
  return layer(inputs, training=training)


@keras_export(v1=['keras.__internal__.legacy.layers.Flatten'])
@tf_export(v1=['layers.Flatten'])
class Flatten(keras_layers.Flatten, base.Layer):
  """Flattens an input tensor while preserving the batch axis (axis 0).

  Args:
    data_format: A string, one of `channels_last` (default) or `channels_first`.
      The ordering of the dimensions in the inputs.
      `channels_last` corresponds to inputs with shape
      `(batch, ..., channels)` while `channels_first` corresponds to
      inputs with shape `(batch, channels, ...)`.

  Examples:

  ```
    x = tf.compat.v1.placeholder(shape=(None, 4, 4), dtype='float32')
    y = Flatten()(x)
    # now `y` has shape `(None, 16)`

    x = tf.compat.v1.placeholder(shape=(None, 3, None), dtype='float32')
    y = Flatten()(x)
    # now `y` has shape `(None, None)`
  ```

  @compatibility(TF2)
  This API is a legacy api that is only compatible with eager execution and
  `tf.function` if you combine it with
  `tf.compat.v1.keras.utils.track_tf1_style_variables`

  Please refer to [tf.layers model mapping section of the migration guide]
  (https://www.tensorflow.org/guide/migrate/model_mapping)
  to learn how to use your TensorFlow v1 model in TF2 with Keras.

  The corresponding TensorFlow v2 layer is `tf.keras.layers.Flatten`.


  #### Structural Mapping to Native TF2

  None of the supported arguments have changed name.

  Before:

  ```python
   flatten = tf.compat.v1.layers.Flatten()
  ```

  After:

  ```python
   flatten = tf.keras.layers.Flatten()
  ```
  @end_compatibility
  """
  pass


@keras_export(v1=['keras.__internal__.legacy.layers.flatten'])
@tf_export(v1=['layers.flatten'])
def flatten(inputs, name=None, data_format='channels_last'):
  """Flattens an input tensor while preserving the batch axis (axis 0).

  Args:
    inputs: Tensor input.
    name: The name of the layer (string).
    data_format: A string, one of `channels_last` (default) or `channels_first`.
      The ordering of the dimensions in the inputs.
      `channels_last` corresponds to inputs with shape
      `(batch, height, width, channels)` while `channels_first` corresponds to
      inputs with shape `(batch, channels, height, width)`.

  Returns:
    Reshaped tensor.

  Examples:

  ```
    x = tf.compat.v1.placeholder(shape=(None, 4, 4), dtype='float32')
    y = flatten(x)
    # now `y` has shape `(None, 16)`

    x = tf.compat.v1.placeholder(shape=(None, 3, None), dtype='float32')
    y = flatten(x)
    # now `y` has shape `(None, None)`
  ```

  @compatibility(TF2)
  This API is a legacy api that is only compatible with eager execution and
  `tf.function` if you combine it with
  `tf.compat.v1.keras.utils.track_tf1_style_variables`

  Please refer to [tf.layers model mapping section of the migration guide]
  (https://www.tensorflow.org/guide/migrate/model_mapping)
  to learn how to use your TensorFlow v1 model in TF2 with Keras.

  The corresponding TensorFlow v2 layer is `tf.keras.layers.Flatten`.


  #### Structural Mapping to Native TF2

  None of the supported arguments have changed name.

  Before:

  ```python
   y = tf.compat.v1.layers.flatten(x)
  ```

  After:

  To migrate code using TF1 functional layers use the [Keras Functional API]
  (https://www.tensorflow.org/guide/keras/functional):

  ```python
   x = tf.keras.Input((28, 28, 1))
   y = tf.keras.layers.Flatten()(x)
   model = tf.keras.Model(x, y)
  ```
  @end_compatibility
  """
  warnings.warn(
      '`tf.layers.flatten` is deprecated and '
      'will be removed in a future version. '
      'Please use `tf.keras.layers.Flatten` instead.',
      stacklevel=2)
  layer = Flatten(name=name, data_format=data_format)
  return layer(inputs)


# Aliases

FullyConnected = Dense
fully_connected = dense
