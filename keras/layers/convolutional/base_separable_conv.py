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
"""Keras abstract base layer for separable nD convolution."""
# pylint: disable=g-classes-have-attributes

from keras import activations
from keras import constraints
from keras import initializers
from keras import regularizers
from keras.engine.input_spec import InputSpec
from keras.layers.convolutional.base_conv import Conv
import tensorflow.compat.v2 as tf


class SeparableConv(Conv):
    """Abstract base layer for separable nD convolution.

    This layer performs a depthwise convolution that acts separately on
    channels, followed by a pointwise convolution that mixes channels.
    If `use_bias` is True and a bias initializer is provided,
    it adds a bias vector to the output.
    It then optionally applies an activation function to produce the final output.

    Args:
      rank: An integer, the rank of the convolution, e.g. "2" for 2D convolution.
      filters: Integer, the dimensionality of the output space (i.e. the number
        of filters in the convolution).
      kernel_size: A tuple or list of integers specifying the spatial
        dimensions of the filters. Can be a single integer to specify the same
        value for all spatial dimensions.
      strides: A tuple or list of integers specifying the strides
        of the convolution. Can be a single integer to specify the same value for
        all spatial dimensions.
        Specifying any `stride` value != 1 is incompatible with specifying
        any `dilation_rate` value != 1.
      padding: One of `"valid"` or `"same"` (case-insensitive).
        `"valid"` means no padding. `"same"` results in padding with zeros evenly
        to the left/right or up/down of the input such that output has the same
        height/width dimension as the input.
      data_format: A string, one of `channels_last` (default) or `channels_first`.
        The ordering of the dimensions in the inputs.
        `channels_last` corresponds to inputs with shape
        `(batch_size, ..., channels)` while `channels_first` corresponds to
        inputs with shape `(batch_size, channels, ...)`.
      dilation_rate: An integer or tuple/list of 2 integers, specifying
        the dilation rate to use for dilated convolution.
        Can be a single integer to specify the same value for
        all spatial dimensions.
        Currently, specifying any `dilation_rate` value != 1 is
        incompatible with specifying any stride value != 1.
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
        convolution kernel.
      pointwise_regularizer: Optional regularizer for the pointwise
        convolution kernel.
      bias_regularizer: Optional regularizer for the bias vector.
      activity_regularizer: Optional regularizer function for the output.
      depthwise_constraint: Optional projection function to be applied to the
        depthwise kernel after being updated by an `Optimizer` (e.g. used for
        norm constraints or value constraints for layer weights). The function
        must take as input the unprojected variable and must return the
        projected variable (which must have the same shape). Constraints are
        not safe to use when doing asynchronous distributed training.
      pointwise_constraint: Optional projection function to be applied to the
        pointwise kernel after being updated by an `Optimizer`.
      bias_constraint: Optional projection function to be applied to the
        bias after being updated by an `Optimizer`.
      trainable: Boolean, if `True` the weights of this layer will be marked as
        trainable (and listed in `layer.trainable_weights`).
    """

    def __init__(
        self,
        rank,
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
        trainable=True,
        name=None,
        **kwargs,
    ):
        super().__init__(
            rank=rank,
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            dilation_rate=dilation_rate,
            activation=activations.get(activation),
            use_bias=use_bias,
            bias_initializer=initializers.get(bias_initializer),
            bias_regularizer=regularizers.get(bias_regularizer),
            activity_regularizer=regularizers.get(activity_regularizer),
            bias_constraint=bias_constraint,
            trainable=trainable,
            name=name,
            **kwargs,
        )
        self.depth_multiplier = depth_multiplier
        self.depthwise_initializer = initializers.get(depthwise_initializer)
        self.pointwise_initializer = initializers.get(pointwise_initializer)
        self.depthwise_regularizer = regularizers.get(depthwise_regularizer)
        self.pointwise_regularizer = regularizers.get(pointwise_regularizer)
        self.depthwise_constraint = constraints.get(depthwise_constraint)
        self.pointwise_constraint = constraints.get(pointwise_constraint)

    def build(self, input_shape):
        input_shape = tf.TensorShape(input_shape)
        channel_axis = self._get_channel_axis()
        if input_shape.dims[channel_axis].value is None:
            raise ValueError(
                "The channel dimension of the inputs should be defined. "
                f"The input_shape received is {input_shape}, "
                f"where axis {channel_axis} (0-based) "
                "is the channel dimension, which found to be `None`."
            )
        input_dim = int(input_shape[channel_axis])
        self.input_spec = InputSpec(
            ndim=self.rank + 2, axes={channel_axis: input_dim}
        )
        depthwise_kernel_shape = self.kernel_size + (
            input_dim,
            self.depth_multiplier,
        )
        pointwise_kernel_shape = (1,) * self.rank + (
            self.depth_multiplier * input_dim,
            self.filters,
        )

        self.depthwise_kernel = self.add_weight(
            name="depthwise_kernel",
            shape=depthwise_kernel_shape,
            initializer=self.depthwise_initializer,
            regularizer=self.depthwise_regularizer,
            constraint=self.depthwise_constraint,
            trainable=True,
            dtype=self.dtype,
        )
        self.pointwise_kernel = self.add_weight(
            name="pointwise_kernel",
            shape=pointwise_kernel_shape,
            initializer=self.pointwise_initializer,
            regularizer=self.pointwise_regularizer,
            constraint=self.pointwise_constraint,
            trainable=True,
            dtype=self.dtype,
        )
        if self.use_bias:
            self.bias = self.add_weight(
                name="bias",
                shape=(self.filters,),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                trainable=True,
                dtype=self.dtype,
            )
        else:
            self.bias = None
        self.built = True

    def call(self, inputs):
        raise NotImplementedError

    def get_config(self):
        config = {
            "filters": self.filters,
            "kernel_size": self.kernel_size,
            "strides": self.strides,
            "padding": self.padding,
            "data_format": self.data_format,
            "depth_multiplier": self.depth_multiplier,
            "dilation_rate": self.dilation_rate,
            "activation": activations.serialize(self.activation),
            "use_bias": self.use_bias,
            "depthwise_initializer": initializers.serialize(
                self.depthwise_initializer
            ),
            "pointwise_initializer": initializers.serialize(
                self.pointwise_initializer
            ),
            "bias_initializer": initializers.serialize(self.bias_initializer),
            "depthwise_regularizer": regularizers.serialize(
                self.depthwise_regularizer
            ),
            "pointwise_regularizer": regularizers.serialize(
                self.pointwise_regularizer
            ),
            "bias_regularizer": regularizers.serialize(self.bias_regularizer),
            "activity_regularizer": regularizers.serialize(
                self.activity_regularizer
            ),
            "depthwise_constraint": constraints.serialize(
                self.depthwise_constraint
            ),
            "pointwise_constraint": constraints.serialize(
                self.pointwise_constraint
            ),
            "bias_constraint": constraints.serialize(self.bias_constraint),
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))
