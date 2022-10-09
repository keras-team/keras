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
"""Keras 1D transposed convolution layer (sometimes called deconvolution)."""


import tensorflow.compat.v2 as tf

from keras import activations
from keras import constraints
from keras import initializers
from keras import regularizers
from keras.dtensor import utils
from keras.engine.input_spec import InputSpec
from keras.layers.convolutional.conv1d import Conv1D
from keras.utils import conv_utils

# isort: off
from tensorflow.python.util.tf_export import keras_export


@keras_export(
    "keras.layers.Conv1DTranspose", "keras.layers.Convolution1DTranspose"
)
class Conv1DTranspose(Conv1D):
    """Transposed convolution layer (sometimes called Deconvolution).

    The need for transposed convolutions generally arises
    from the desire to use a transformation going in the opposite direction
    of a normal convolution, i.e., from something that has the shape of the
    output of some convolution to something that has the shape of its input
    while maintaining a connectivity pattern that is compatible with
    said convolution.

    When using this layer as the first layer in a model,
    provide the keyword argument `input_shape`
    (tuple of integers or `None`, does not include the sample axis),
    e.g. `input_shape=(128, 3)` for data with 128 time steps and 3 channels.

    Args:
      filters: Integer, the dimensionality of the output space
        (i.e. the number of output filters in the convolution).
      kernel_size: An integer length of the 1D convolution window.
      strides: An integer specifying the stride of the convolution along the
        time dimension. Specifying a stride value != 1 is incompatible with
        specifying a `dilation_rate` value != 1. Defaults to 1.
      padding: one of `"valid"` or `"same"` (case-insensitive).
        `"valid"` means no padding. `"same"` results in padding with zeros
        evenly to the left/right or up/down of the input such that output has
        the same height/width dimension as the input.
      output_padding: An integer specifying the amount of padding along
        the time dimension of the output tensor.
        The amount of output padding must be lower than the stride.
        If set to `None` (default), the output shape is inferred.
      data_format: A string, one of `channels_last` (default) or
        `channels_first`.  The ordering of the dimensions in the inputs.
        `channels_last` corresponds to inputs with shape
        `(batch_size, length, channels)` while `channels_first` corresponds to
        inputs with shape `(batch_size, channels, length)`.
      dilation_rate: an integer, specifying
        the dilation rate to use for dilated convolution.
        Currently, specifying a `dilation_rate` value != 1 is
        incompatible with specifying a stride value != 1.
        Also dilation rate larger than 1 is not currently supported.
      activation: Activation function to use.
        If you don't specify anything, no activation is applied
        (see `keras.activations`).
      use_bias: Boolean, whether the layer uses a bias vector.
      kernel_initializer: Initializer for the `kernel` weights matrix
        (see `keras.initializers`). Defaults to 'glorot_uniform'.
      bias_initializer: Initializer for the bias vector
        (see `keras.initializers`). Defaults to 'zeros'.
      kernel_regularizer: Regularizer function applied to
        the `kernel` weights matrix (see `keras.regularizers`).
      bias_regularizer: Regularizer function applied to the bias vector
        (see `keras.regularizers`).
      activity_regularizer: Regularizer function applied to
        the output of the layer (its "activation") (see `keras.regularizers`).
      kernel_constraint: Constraint function applied to the kernel matrix
        (see `keras.constraints`).
      bias_constraint: Constraint function applied to the bias vector
        (see `keras.constraints`).

    Input shape:
      3D tensor with shape:
      `(batch_size, steps, channels)`

    Output shape:
      3D tensor with shape:
      `(batch_size, new_steps, filters)`
      If `output_padding` is specified:
      ```
      new_timesteps = ((timesteps - 1) * strides + kernel_size -
      2 * padding + output_padding)
      ```

    Returns:
      A tensor of rank 3 representing
      `activation(conv1dtranspose(inputs, kernel) + bias)`.

    Raises:
      ValueError: if `padding` is "causal".
      ValueError: when both `strides` > 1 and `dilation_rate` > 1.

    References:
      - [A guide to convolution arithmetic for deep learning](
        https://arxiv.org/abs/1603.07285v1)
      - [Deconvolutional Networks](
        https://www.matthewzeiler.com/mattzeiler/deconvolutionalnetworks.pdf)
    """

    @utils.allow_initializer_layout
    def __init__(
        self,
        filters,
        kernel_size,
        strides=1,
        padding="valid",
        output_padding=None,
        data_format=None,
        dilation_rate=1,
        activation=None,
        use_bias=True,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        **kwargs,
    ):
        super().__init__(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            dilation_rate=dilation_rate,
            activation=activations.get(activation),
            use_bias=use_bias,
            kernel_initializer=initializers.get(kernel_initializer),
            bias_initializer=initializers.get(bias_initializer),
            kernel_regularizer=regularizers.get(kernel_regularizer),
            bias_regularizer=regularizers.get(bias_regularizer),
            activity_regularizer=regularizers.get(activity_regularizer),
            kernel_constraint=constraints.get(kernel_constraint),
            bias_constraint=constraints.get(bias_constraint),
            **kwargs,
        )

        self.output_padding = output_padding
        if self.output_padding is not None:
            self.output_padding = conv_utils.normalize_tuple(
                self.output_padding, 1, "output_padding", allow_zero=True
            )
            for stride, out_pad in zip(self.strides, self.output_padding):
                if out_pad >= stride:
                    raise ValueError(
                        "Strides must be greater than output padding. "
                        f"Received strides={self.strides}, "
                        f"output_padding={self.output_padding}."
                    )

    def build(self, input_shape):
        input_shape = tf.TensorShape(input_shape)
        if len(input_shape) != 3:
            raise ValueError(
                "Inputs should have rank 3. "
                f"Received input_shape={input_shape}."
            )
        channel_axis = self._get_channel_axis()
        if input_shape.dims[channel_axis].value is None:
            raise ValueError(
                "The channel dimension of the inputs "
                "to `Conv1DTranspose` should be defined. "
                f"The input_shape received is {input_shape}, "
                f"where axis {channel_axis} (0-based) "
                "is the channel dimension, which found to be `None`."
            )
        input_dim = int(input_shape[channel_axis])
        self.input_spec = InputSpec(ndim=3, axes={channel_axis: input_dim})
        kernel_shape = self.kernel_size + (self.filters, input_dim)

        self.kernel = self.add_weight(
            name="kernel",
            shape=kernel_shape,
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
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
        inputs_shape = tf.shape(inputs)
        batch_size = inputs_shape[0]
        if self.data_format == "channels_first":
            t_axis = 2
        else:
            t_axis = 1

        length = inputs_shape[t_axis]
        if self.output_padding is None:
            output_padding = None
        else:
            output_padding = self.output_padding[0]

        # Infer the dynamic output shape:
        out_length = conv_utils.deconv_output_length(
            length,
            self.kernel_size[0],
            padding=self.padding,
            output_padding=output_padding,
            stride=self.strides[0],
            dilation=self.dilation_rate[0],
        )
        if self.data_format == "channels_first":
            output_shape = (batch_size, self.filters, out_length)
        else:
            output_shape = (batch_size, out_length, self.filters)
        data_format = conv_utils.convert_data_format(self.data_format, ndim=3)

        output_shape_tensor = tf.stack(output_shape)
        outputs = tf.nn.conv1d_transpose(
            inputs,
            self.kernel,
            output_shape_tensor,
            strides=self.strides,
            padding=self.padding.upper(),
            data_format=data_format,
            dilations=self.dilation_rate,
        )

        if not tf.executing_eagerly() and inputs.shape.rank:
            # Infer the static output shape:
            out_shape = self.compute_output_shape(inputs.shape)
            outputs.set_shape(out_shape)

        if self.use_bias:
            outputs = tf.nn.bias_add(
                outputs, self.bias, data_format=data_format
            )

        if self.activation is not None:
            return self.activation(outputs)
        return outputs

    def compute_output_shape(self, input_shape):
        input_shape = tf.TensorShape(input_shape).as_list()
        output_shape = list(input_shape)
        if self.data_format == "channels_first":
            c_axis, t_axis = 1, 2
        else:
            c_axis, t_axis = 2, 1

        if self.output_padding is None:
            output_padding = None
        else:
            output_padding = self.output_padding[0]
        output_shape[c_axis] = self.filters
        output_shape[t_axis] = conv_utils.deconv_output_length(
            output_shape[t_axis],
            self.kernel_size[0],
            padding=self.padding,
            output_padding=output_padding,
            stride=self.strides[0],
            dilation=self.dilation_rate[0],
        )
        return tf.TensorShape(output_shape)

    def get_config(self):
        config = super().get_config()
        config["output_padding"] = self.output_padding
        return config


# Alias

Convolution1DTranspose = Conv1DTranspose
