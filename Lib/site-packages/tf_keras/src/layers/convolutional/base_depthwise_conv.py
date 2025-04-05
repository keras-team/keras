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
"""Keras abstract base for depthwise convolutions."""


import tensorflow.compat.v2 as tf

from tf_keras.src import constraints
from tf_keras.src import initializers
from tf_keras.src import regularizers
from tf_keras.src.engine.input_spec import InputSpec
from tf_keras.src.layers.convolutional.base_conv import Conv


class DepthwiseConv(Conv):
    """Depthwise convolution.

    Depthwise convolution is a type of convolution in which each input channel
    is convolved with a different kernel (called a depthwise kernel). You can
    understand depthwise convolution as the first step in a depthwise separable
    convolution.

    It is implemented via the following steps:

    - Split the input into individual channels.
    - Convolve each channel with an individual depthwise kernel with
      `depth_multiplier` output channels.
    - Concatenate the convolved outputs along the channels axis.

    Unlike a regular convolution, depthwise convolution does not mix
    information across different input channels.

    The `depth_multiplier` argument determines how many filter are applied to
    one input channel. As such, it controls the amount of output channels that
    are generated per input channel in the depthwise step.

    Args:
      kernel_size: A tuple or list of integers specifying the spatial dimensions
        of the filters. Can be a single integer to specify the same value for
        all spatial dimensions.
      strides: A tuple or list of integers specifying the strides of the
        convolution. Can be a single integer to specify the same value for all
        spatial dimensions. Specifying any `stride` value != 1 is incompatible
        with specifying any `dilation_rate` value != 1.
      padding: One of `"valid"` or `"same"` (case-insensitive). `"valid"` means
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
        shape `(batch_size, channels, height, width)`. If left unspecified,
        uses `image_data_format` value found in your TF-Keras config file at
        `~/.keras/keras.json` (if exists) else 'channels_last'.
        Defaults to 'channels_last'.
      dilation_rate: An integer or tuple/list of 2 integers, specifying the
        dilation rate to use for dilated convolution. Currently, specifying any
        `dilation_rate` value != 1 is incompatible with specifying any `strides`
        value != 1.
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
        rank,
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
        **kwargs,
    ):
        super().__init__(
            rank,
            filters=None,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            dilation_rate=dilation_rate,
            activation=activation,
            use_bias=use_bias,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            bias_constraint=bias_constraint,
            **kwargs,
        )
        self.depth_multiplier = depth_multiplier
        self.depthwise_initializer = initializers.get(depthwise_initializer)
        self.depthwise_regularizer = regularizers.get(depthwise_regularizer)
        self.depthwise_constraint = constraints.get(depthwise_constraint)
        self.bias_initializer = initializers.get(bias_initializer)

    def build(self, input_shape):
        if len(input_shape) != self.rank + 2:
            raise ValueError(
                "Inputs to `DepthwiseConv` should have "
                f"rank {self.rank + 2}. "
                f"Received input_shape={input_shape}."
            )
        input_shape = tf.TensorShape(input_shape)
        channel_axis = self._get_channel_axis()
        if input_shape.dims[channel_axis].value is None:
            raise ValueError(
                "The channel dimension of the inputs to `DepthwiseConv` "
                "should be defined. "
                f"The input_shape received is {input_shape}, "
                f"where axis {channel_axis} (0-based) "
                "is the channel dimension, which found to be `None`."
            )
        input_dim = int(input_shape[channel_axis])
        depthwise_kernel_shape = self.kernel_size + (
            input_dim,
            self.depth_multiplier,
        )

        self.depthwise_kernel = self.add_weight(
            shape=depthwise_kernel_shape,
            initializer=self.depthwise_initializer,
            name="depthwise_kernel",
            regularizer=self.depthwise_regularizer,
            constraint=self.depthwise_constraint,
        )

        if self.use_bias:
            self.bias = self.add_weight(
                shape=(input_dim * self.depth_multiplier,),
                initializer=self.bias_initializer,
                name="bias",
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
            )
        else:
            self.bias = None
        # Set input spec.
        self.input_spec = InputSpec(
            min_ndim=self.rank + 2, axes={channel_axis: input_dim}
        )
        self.built = True

    def call(self, inputs):
        raise NotImplementedError

    def get_config(self):
        config = super().get_config()
        config.pop("filters")
        config.pop("kernel_initializer")
        config.pop("kernel_regularizer")
        config.pop("kernel_constraint")
        config["depth_multiplier"] = self.depth_multiplier
        config["depthwise_initializer"] = initializers.serialize(
            self.depthwise_initializer
        )
        config["depthwise_regularizer"] = regularizers.serialize(
            self.depthwise_regularizer
        )
        config["depthwise_constraint"] = constraints.serialize(
            self.depthwise_constraint
        )
        return config

