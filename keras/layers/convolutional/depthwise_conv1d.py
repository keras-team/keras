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
"""Keras depthwise 1D convolution."""


import tensorflow.compat.v2 as tf

from keras.layers.convolutional.base_depthwise_conv import DepthwiseConv
from keras.utils import conv_utils
from keras.utils import tf_utils

# isort: off
from tensorflow.python.util.tf_export import keras_export


@keras_export("keras.layers.DepthwiseConv1D")
class DepthwiseConv1D(DepthwiseConv):
    """Depthwise 1D convolution.

    Depthwise convolution is a type of convolution in which each input channel
    is convolved with a different kernel (called a depthwise kernel). You can
    understand depthwise convolution as the first step in a depthwise separable
    convolution.

    It is implemented via the following steps:

    - Split the input into individual channels.
    - Convolve each channel with an individual depthwise kernel with
      `depth_multiplier` output channels.
    - Concatenate the convolved outputs along the channels axis.

    Unlike a regular 1D convolution, depthwise convolution does not mix
    information across different input channels.

    The `depth_multiplier` argument determines how many filter are applied to
    one input channel. As such, it controls the amount of output channels that
    are generated per input channel in the depthwise step.

    Args:
      kernel_size: An integer, specifying the height and width of the 1D
        convolution window. Can be a single integer to specify the same value
        for all spatial dimensions.
      strides: An integer, specifying the strides of the convolution along the
        height and width. Can be a single integer to specify the same value for
        all spatial dimensions. Specifying any stride value != 1 is incompatible
        with specifying any `dilation_rate` value != 1.
      padding: one of `'valid'` or `'same'` (case-insensitive). `"valid"` means
        no padding. `"same"` results in padding with zeros evenly to the
        left/right or up/down of the input such that output has the same
        height/width dimension as the input.
      depth_multiplier: The number of depthwise convolution output channels for
        each input channel. The total number of depthwise convolution output
        channels will be equal to `filters_in * depth_multiplier`.
      data_format: A string, one of `channels_last` (default) or
        `channels_first`.  The ordering of the dimensions in the inputs.
        `channels_last` corresponds to inputs with shape `(batch_size, height,
        width, channels)` while `channels_first` corresponds to inputs with
        shape `(batch_size, channels, height, width)`. It defaults to the
        `image_data_format` value found in your Keras config file at
        `~/.keras/keras.json`. If you never set it, then it will be
        'channels_last'.
      dilation_rate: A single integer, specifying the dilation rate to use for
        dilated convolution. Currently, specifying any `dilation_rate`
        value != 1 is incompatible with specifying any stride value != 1.
      activation: Activation function to use. If you don't specify anything, no
        activation is applied (see `keras.activations`).
      use_bias: Boolean, whether the layer uses a bias vector.
      depthwise_initializer: Initializer for the depthwise kernel matrix (see
        `keras.initializers`). If None, the default initializer
        ('glorot_uniform') will be used.
      bias_initializer: Initializer for the bias vector (see
        `keras.initializers`). If None, the default initializer ('zeros') will
        be used.
      depthwise_regularizer: Regularizer function applied to the depthwise
        kernel matrix (see `keras.regularizers`).
      bias_regularizer: Regularizer function applied to the bias vector (see
        `keras.regularizers`).
      activity_regularizer: Regularizer function applied to the output of the
        layer (its 'activation') (see `keras.regularizers`).
      depthwise_constraint: Constraint function applied to the depthwise kernel
        matrix (see `keras.constraints`).
      bias_constraint: Constraint function applied to the bias vector (see
        `keras.constraints`).

    Input shape:
      4D tensor with shape: `[batch_size, channels, rows, cols]` if
        data_format='channels_first'
      or 4D tensor with shape: `[batch_size, rows, cols, channels]` if
        data_format='channels_last'.

    Output shape:
      4D tensor with shape: `[batch_size, channels * depth_multiplier, new_rows,
        new_cols]` if `data_format='channels_first'`
        or 4D tensor with shape: `[batch_size,
        new_rows, new_cols, channels * depth_multiplier]` if
        `data_format='channels_last'`. `rows` and `cols` values might have
        changed due to padding.

    Returns:
      A tensor of rank 4 representing
      `activation(depthwiseconv2d(inputs, kernel) + bias)`.

    Raises:
      ValueError: if `padding` is "causal".
      ValueError: when both `strides` > 1 and `dilation_rate` > 1.
    """

    def __init__(
        self,
        kernel_size,
        strides=1,
        padding="valid",
        depth_multiplier=1,
        data_format=None,
        dilation_rate=1,
        activation=None,
        use_bias=True,
        depthwise_initializer="glorot_uniform",
        bias_initializer="zeros",
        depthwise_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        depthwise_constraint=None,
        bias_constraint=None,
        **kwargs
    ):
        super().__init__(
            1,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            depth_multiplier=depth_multiplier,
            data_format=data_format,
            dilation_rate=dilation_rate,
            activation=activation,
            use_bias=use_bias,
            depthwise_initializer=depthwise_initializer,
            bias_initializer=bias_initializer,
            depthwise_regularizer=depthwise_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            depthwise_constraint=depthwise_constraint,
            bias_constraint=bias_constraint,
            **kwargs
        )

    def call(self, inputs):
        if self.data_format == "channels_last":
            strides = (1,) + self.strides * 2 + (1,)
            spatial_start_dim = 1
        else:
            strides = (1, 1) + self.strides * 2
            spatial_start_dim = 2
        inputs = tf.expand_dims(inputs, spatial_start_dim)
        depthwise_kernel = tf.expand_dims(self.depthwise_kernel, axis=0)
        dilation_rate = (1,) + self.dilation_rate

        outputs = tf.nn.depthwise_conv2d(
            inputs,
            depthwise_kernel,
            strides=strides,
            padding=self.padding.upper(),
            dilations=dilation_rate,
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

    @tf_utils.shape_type_conversion
    def compute_output_shape(self, input_shape):
        if self.data_format == "channels_first":
            rows = input_shape[2]
            out_filters = input_shape[1] * self.depth_multiplier
        elif self.data_format == "channels_last":
            rows = input_shape[1]
            out_filters = input_shape[2] * self.depth_multiplier

        rows = conv_utils.conv_output_length(
            rows,
            self.kernel_size[0],
            self.padding,
            self.strides[0],
            self.dilation_rate[0],
        )
        if self.data_format == "channels_first":
            return (input_shape[0], out_filters, rows)
        elif self.data_format == "channels_last":
            return (input_shape[0], rows, out_filters)
