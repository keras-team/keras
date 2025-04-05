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
"""Keras depthwise separable 2D convolution."""


import tensorflow.compat.v2 as tf

from tf_keras.src import activations
from tf_keras.src import constraints
from tf_keras.src import initializers
from tf_keras.src import regularizers
from tf_keras.src.layers.convolutional.base_separable_conv import SeparableConv
from tf_keras.src.utils import conv_utils

# isort: off
from tensorflow.python.util.tf_export import keras_export


@keras_export(
    "keras.layers.SeparableConv2D", "keras.layers.SeparableConvolution2D"
)
class SeparableConv2D(SeparableConv):
    """Depthwise separable 2D convolution.

    Separable convolutions consist of first performing
    a depthwise spatial convolution
    (which acts on each input channel separately)
    followed by a pointwise convolution which mixes the resulting
    output channels. The `depth_multiplier` argument controls how many
    output channels are generated per input channel in the depthwise step.

    Intuitively, separable convolutions can be understood as
    a way to factorize a convolution kernel into two smaller kernels,
    or as an extreme version of an Inception block.

    Args:
      filters: Integer, the dimensionality of the output space
        (i.e. the number of output filters in the convolution).
      kernel_size: An integer or tuple/list of 2 integers, specifying the
        height and width of the 2D convolution window.
        Can be a single integer to specify the same value for
        all spatial dimensions.
      strides: An integer or tuple/list of 2 integers,
        specifying the strides of the convolution along the height and width.
        Can be a single integer to specify the same value for
        all spatial dimensions. Current implementation only supports equal
        length strides in the row and column dimensions.
        Specifying any stride value != 1 is incompatible with specifying
        any `dilation_rate` value != 1.
      padding: one of `"valid"` or `"same"` (case-insensitive).
        `"valid"` means no padding. `"same"` results in padding with zeros
        evenly to the left/right or up/down of the input such that output has
        the same height/width dimension as the input.
      data_format: A string,
        one of `channels_last` (default) or `channels_first`.
        The ordering of the dimensions in the inputs.
        `channels_last` corresponds to inputs with shape
        `(batch_size, height, width, channels)` while `channels_first`
        corresponds to inputs with shape
        `(batch_size, channels, height, width)`.
        When unspecified, uses `image_data_format` value found in your Keras
        config file at `~/.keras/keras.json` (if exists) else 'channels_last'.
        Defaults to 'channels_last'.
      dilation_rate: An integer or tuple/list of 2 integers, specifying
        the dilation rate to use for dilated convolution.
      depth_multiplier: The number of depthwise convolution output channels
        for each input channel.
        The total number of depthwise convolution output
        channels will be equal to `filters_in * depth_multiplier`.
      activation: Activation function to use.
        If you don't specify anything, no activation is applied
        (see `keras.activations`).
      use_bias: Boolean, whether the layer uses a bias vector.
      depthwise_initializer: An initializer for the depthwise convolution kernel
        (see `keras.initializers`). If None, then the default initializer
        ('glorot_uniform') will be used.
      pointwise_initializer: An initializer for the pointwise convolution kernel
        (see `keras.initializers`). If None, then the default initializer
        ('glorot_uniform') will be used.
      bias_initializer: An initializer for the bias vector. If None, the default
        initializer ('zeros') will be used (see `keras.initializers`).
      depthwise_regularizer: Regularizer function applied to
        the depthwise kernel matrix (see `keras.regularizers`).
      pointwise_regularizer: Regularizer function applied to
        the pointwise kernel matrix (see `keras.regularizers`).
      bias_regularizer: Regularizer function applied to the bias vector
        (see `keras.regularizers`).
      activity_regularizer: Regularizer function applied to
        the output of the layer (its "activation")
        (see `keras.regularizers`).
      depthwise_constraint: Constraint function applied to
        the depthwise kernel matrix
        (see `keras.constraints`).
      pointwise_constraint: Constraint function applied to
        the pointwise kernel matrix
        (see `keras.constraints`).
      bias_constraint: Constraint function applied to the bias vector
        (see `keras.constraints`).

    Input shape:
      4D tensor with shape:
      `(batch_size, channels, rows, cols)` if data_format='channels_first'
      or 4D tensor with shape:
      `(batch_size, rows, cols, channels)` if data_format='channels_last'.

    Output shape:
      4D tensor with shape:
      `(batch_size, filters, new_rows, new_cols)` if
      data_format='channels_first'
      or 4D tensor with shape:
      `(batch_size, new_rows, new_cols, filters)` if
      data_format='channels_last'.  `rows` and `cols` values might have changed
      due to padding.

    Returns:
      A tensor of rank 4 representing
      `activation(separableconv2d(inputs, kernel) + bias)`.

    Raises:
      ValueError: if `padding` is "causal".
    """

    def __init__(
        self,
        filters,
        kernel_size,
        strides=(1, 1),
        padding="valid",
        data_format=None,
        dilation_rate=(1, 1),
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
            rank=2,
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
        # Apply the actual ops.
        if self.data_format == "channels_last":
            strides = (1,) + self.strides + (1,)
        else:
            strides = (1, 1) + self.strides
        outputs = tf.nn.separable_conv2d(
            inputs,
            self.depthwise_kernel,
            self.pointwise_kernel,
            strides=strides,
            padding=self.padding.upper(),
            dilations=self.dilation_rate,
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

        if self.activation is not None:
            return self.activation(outputs)
        return outputs


# Alias

SeparableConvolution2D = SeparableConv2D

