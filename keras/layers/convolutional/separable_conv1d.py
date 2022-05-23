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
"""Keras depthwise separable 1D convolution."""
# pylint: disable=g-classes-have-attributes,g-direct-tensorflow-import

from keras import activations
from keras import constraints
from keras import initializers
from keras import regularizers
from keras.layers.convolutional.base_separable_conv import SeparableConv
from keras.utils import conv_utils
import tensorflow.compat.v2 as tf

from tensorflow.python.util.tf_export import keras_export


@keras_export(
    "keras.layers.SeparableConv1D", "keras.layers.SeparableConvolution1D"
)
class SeparableConv1D(SeparableConv):
    """Depthwise separable 1D convolution.

    This layer performs a depthwise convolution that acts separately on
    channels, followed by a pointwise convolution that mixes channels.
    If `use_bias` is True and a bias initializer is provided,
    it adds a bias vector to the output.
    It then optionally applies an activation function to produce the final output.

    Args:
      filters: Integer, the dimensionality of the output space (i.e. the number
        of filters in the convolution).
      kernel_size: A single integer specifying the spatial
        dimensions of the filters.
      strides: A single integer specifying the strides
        of the convolution.
        Specifying any `stride` value != 1 is incompatible with specifying
        any `dilation_rate` value != 1.
      padding: One of `"valid"`, `"same"`, or `"causal"` (case-insensitive).
        `"valid"` means no padding. `"same"` results in padding with zeros evenly
        to the left/right or up/down of the input such that output has the same
        height/width dimension as the input. `"causal"` results in causal
        (dilated) convolutions, e.g. `output[t]` does not depend on `input[t+1:]`.
      data_format: A string, one of `channels_last` (default) or `channels_first`.
        The ordering of the dimensions in the inputs.
        `channels_last` corresponds to inputs with shape
        `(batch_size, length, channels)` while `channels_first` corresponds to
        inputs with shape `(batch_size, channels, length)`.
      dilation_rate: A single integer, specifying
        the dilation rate to use for dilated convolution.
      depth_multiplier: The number of depthwise convolution output channels for
        each input channel. The total number of depthwise convolution output
        channels will be equal to `num_filters_in * depth_multiplier`.
      activation: Activation function to use.
        If you don't specify anything, no activation is applied
        (see `keras.activations`).
      use_bias: Boolean, whether the layer uses a bias.
      depthwise_initializer: An initializer for the depthwise convolution kernel
        (see `keras.initializers`). If None, then the default initializer
        ('glorot_uniform') will be used.
      pointwise_initializer: An initializer for the pointwise convolution kernel
        (see `keras.initializers`). If None, then the default initializer
        ('glorot_uniform') will be used.
      bias_initializer: An initializer for the bias vector. If None, the default
        initializer ('zeros') will be used (see `keras.initializers`).
      depthwise_regularizer: Optional regularizer for the depthwise
        convolution kernel (see `keras.regularizers`).
      pointwise_regularizer: Optional regularizer for the pointwise
        convolution kernel (see `keras.regularizers`).
      bias_regularizer: Optional regularizer for the bias vector
        (see `keras.regularizers`).
      activity_regularizer: Optional regularizer function for the output
        (see `keras.regularizers`).
      depthwise_constraint: Optional projection function to be applied to the
        depthwise kernel after being updated by an `Optimizer` (e.g. used for
        norm constraints or value constraints for layer weights). The function
        must take as input the unprojected variable and must return the
        projected variable (which must have the same shape). Constraints are
        not safe to use when doing asynchronous distributed training
        (see `keras.constraints`).
      pointwise_constraint: Optional projection function to be applied to the
        pointwise kernel after being updated by an `Optimizer`
        (see `keras.constraints`).
      bias_constraint: Optional projection function to be applied to the
        bias after being updated by an `Optimizer`
        (see `keras.constraints`).
      trainable: Boolean, if `True` the weights of this layer will be marked as
        trainable (and listed in `layer.trainable_weights`).

    Input shape:
      3D tensor with shape:
      `(batch_size, channels, steps)` if data_format='channels_first'
      or 3D tensor with shape:
      `(batch_size, steps, channels)` if data_format='channels_last'.

    Output shape:
      3D tensor with shape:
      `(batch_size, filters, new_steps)` if data_format='channels_first'
      or 3D tensor with shape:
      `(batch_size,  new_steps, filters)` if data_format='channels_last'.
      `new_steps` value might have changed due to padding or strides.

    Returns:
      A tensor of rank 3 representing
      `activation(separableconv1d(inputs, kernel) + bias)`.
    """

    def __init__(
        self,
        filters,
        kernel_size,
        strides=1,
        padding="valid",
        data_format=None,
        dilation_rate=1,
        depth_multiplier=1,
        activation=None,
        use_bias=True,
        depthwise_initializer="glorot_uniform",
        pointwise_initializer="glorot_uniform",
        bias_initializer="zeros",
        depthwise_regularizer=None,
        pointwise_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        depthwise_constraint=None,
        pointwise_constraint=None,
        bias_constraint=None,
        **kwargs
    ):
        super().__init__(
            rank=1,
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            dilation_rate=dilation_rate,
            depth_multiplier=depth_multiplier,
            activation=activations.get(activation),
            use_bias=use_bias,
            depthwise_initializer=initializers.get(depthwise_initializer),
            pointwise_initializer=initializers.get(pointwise_initializer),
            bias_initializer=initializers.get(bias_initializer),
            depthwise_regularizer=regularizers.get(depthwise_regularizer),
            pointwise_regularizer=regularizers.get(pointwise_regularizer),
            bias_regularizer=regularizers.get(bias_regularizer),
            activity_regularizer=regularizers.get(activity_regularizer),
            depthwise_constraint=constraints.get(depthwise_constraint),
            pointwise_constraint=constraints.get(pointwise_constraint),
            bias_constraint=constraints.get(bias_constraint),
            **kwargs
        )

    def call(self, inputs):
        if self.padding == "causal":
            inputs = tf.pad(inputs, self._compute_causal_padding(inputs))
        if self.data_format == "channels_last":
            strides = (1,) + self.strides * 2 + (1,)
            spatial_start_dim = 1
        else:
            strides = (1, 1) + self.strides * 2
            spatial_start_dim = 2

        # Explicitly broadcast inputs and kernels to 4D.
        # TODO(fchollet): refactor when a native separable_conv1d op is available.
        inputs = tf.expand_dims(inputs, spatial_start_dim)
        depthwise_kernel = tf.expand_dims(self.depthwise_kernel, 0)
        pointwise_kernel = tf.expand_dims(self.pointwise_kernel, 0)
        dilation_rate = (1,) + self.dilation_rate

        if self.padding == "causal":
            op_padding = "valid"
        else:
            op_padding = self.padding
        outputs = tf.compat.v1.nn.separable_conv2d(
            inputs,
            depthwise_kernel,
            pointwise_kernel,
            strides=strides,
            padding=op_padding.upper(),
            rate=dilation_rate,
            data_format=conv_utils.convert_data_format(
                self.data_format, ndim=4
            ),
        )

        if self.use_bias:
            outputs = tf.nn.bias_add(
                outputs,
                self.bias,
                data_format=conv_utils.convert_data_format(
                    self.data_format, ndim=4
                ),
            )

        outputs = tf.squeeze(outputs, [spatial_start_dim])

        if self.activation is not None:
            return self.activation(outputs)
        return outputs


# Alias

SeparableConvolution1D = SeparableConv1D
