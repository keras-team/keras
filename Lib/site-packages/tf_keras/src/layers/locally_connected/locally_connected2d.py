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

"""Locally-connected layer for 2D input."""

from tf_keras.src import activations
from tf_keras.src import backend
from tf_keras.src import constraints
from tf_keras.src import initializers
from tf_keras.src import regularizers
from tf_keras.src.engine.base_layer import Layer
from tf_keras.src.engine.input_spec import InputSpec
from tf_keras.src.layers.locally_connected import locally_connected_utils
from tf_keras.src.utils import conv_utils
from tf_keras.src.utils import tf_utils

# isort: off
from tensorflow.python.util.tf_export import keras_export


@keras_export("keras.layers.LocallyConnected2D")
class LocallyConnected2D(Layer):
    """Locally-connected layer for 2D inputs.

    The `LocallyConnected2D` layer works similarly
    to the `Conv2D` layer, except that weights are unshared,
    that is, a different set of filters is applied at each
    different patch of the input.

    Note: layer attributes cannot be modified after the layer has been called
    once (except the `trainable` attribute).

    Examples:
    ```python
        # apply a 3x3 unshared weights convolution with 64 output filters on a
        32x32 image
        # with `data_format="channels_last"`:
        model = Sequential()
        model.add(LocallyConnected2D(64, (3, 3), input_shape=(32, 32, 3)))
        # now model.output_shape == (None, 30, 30, 64)
        # notice that this layer will consume (30*30)*(3*3*3*64) + (30*30)*64
        parameters

        # add a 3x3 unshared weights convolution on top, with 32 output filters:
        model.add(LocallyConnected2D(32, (3, 3)))
        # now model.output_shape == (None, 28, 28, 32)
    ```

    Args:
        filters: Integer, the dimensionality of the output space (i.e. the
          number of output filters in the convolution).
        kernel_size: An integer or tuple/list of 2 integers, specifying the
          width and height of the 2D convolution window. Can be a single integer
          to specify the same value for all spatial dimensions.
        strides: An integer or tuple/list of 2 integers, specifying the strides
          of the convolution along the width and height. Can be a single integer
          to specify the same value for all spatial dimensions.
        padding: Currently only support `"valid"` (case-insensitive). `"same"`
          will be supported in future. `"valid"` means no padding.
        data_format: A string, one of `channels_last` (default) or
          `channels_first`. The ordering of the dimensions in the inputs.
          `channels_last` corresponds to inputs with shape `(batch, height,
            width, channels)` while `channels_first` corresponds to inputs with
            shape
          `(batch, channels, height, width)`. When unspecified, uses
          `image_data_format` value found in your TF-Keras config file at
          `~/.keras/keras.json` (if exists) else 'channels_last'.
          Defaults to 'channels_last'.
        activation: Activation function to use. If you don't specify anything,
          no activation is applied (ie. "linear" activation: `a(x) = x`).
        use_bias: Boolean, whether the layer uses a bias vector.
        kernel_initializer: Initializer for the `kernel` weights matrix.
        bias_initializer: Initializer for the bias vector.
        kernel_regularizer: Regularizer function applied to the `kernel` weights
          matrix.
        bias_regularizer: Regularizer function applied to the bias vector.
        activity_regularizer: Regularizer function applied to the output of the
          layer (its "activation").
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
              large `np.prod(input_size)`, `np.prod(output_size)`), and "sparse"
              stands for few connections between inputs and outputs, i.e. small
              ratio `filters * input_filters * np.prod(kernel_size) /
              (np.prod(input_size) * np.prod(strides))`, where inputs to and
              outputs of the layer are assumed to have shapes `input_size +
              (input_filters,)`, `output_size + (filters,)` respectively. It is
              recommended to benchmark each in the setting of interest to pick
              the most efficient one (in terms of speed and memory usage).
              Correct choice of implementation can lead to dramatic speed
              improvements (e.g. 50X), potentially at the expense of RAM. Also,
              only `padding="valid"` is supported by `implementation=1`.
    Input shape:
        4D tensor with shape: `(samples, channels, rows, cols)` if
          data_format='channels_first'
        or 4D tensor with shape: `(samples, rows, cols, channels)` if
          data_format='channels_last'.
    Output shape:
        4D tensor with shape: `(samples, filters, new_rows, new_cols)` if
          data_format='channels_first'
        or 4D tensor with shape: `(samples, new_rows, new_cols, filters)` if
          data_format='channels_last'. `rows` and `cols` values might have
          changed due to padding.
    """

    def __init__(
        self,
        filters,
        kernel_size,
        strides=(1, 1),
        padding="valid",
        data_format=None,
        activation=None,
        use_bias=True,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        implementation=1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.filters = filters
        self.kernel_size = conv_utils.normalize_tuple(
            kernel_size, 2, "kernel_size"
        )
        self.strides = conv_utils.normalize_tuple(
            strides, 2, "strides", allow_zero=True
        )
        self.padding = conv_utils.normalize_padding(padding)
        if self.padding != "valid" and implementation == 1:
            raise ValueError(
                "Invalid border mode for LocallyConnected2D "
                '(only "valid" is supported if implementation is 1): ' + padding
            )
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
        self.input_spec = InputSpec(ndim=4)

    @property
    def _use_input_spec_as_call_signature(self):
        return False

    @tf_utils.shape_type_conversion
    def build(self, input_shape):
        if self.data_format == "channels_last":
            input_row, input_col = input_shape[1:-1]
            input_filter = input_shape[3]
        else:
            input_row, input_col = input_shape[2:]
            input_filter = input_shape[1]
        if input_row is None or input_col is None:
            raise ValueError(
                "The spatial dimensions of the inputs to "
                " a LocallyConnected2D layer "
                "should be fully-defined, but layer received "
                "the inputs shape " + str(input_shape)
            )
        output_row = conv_utils.conv_output_length(
            input_row, self.kernel_size[0], self.padding, self.strides[0]
        )
        output_col = conv_utils.conv_output_length(
            input_col, self.kernel_size[1], self.padding, self.strides[1]
        )
        self.output_row = output_row
        self.output_col = output_col

        if self.output_row <= 0 or self.output_col <= 0:
            raise ValueError(
                "One of the dimensions in the output is <= 0 "
                f"due to downsampling in {self.name}. Consider "
                "increasing the input size. "
                f"Received input shape {input_shape} which would produce "
                "output shape with a zero or negative value in a "
                "dimension."
            )

        if self.implementation == 1:
            self.kernel_shape = (
                output_row * output_col,
                self.kernel_size[0] * self.kernel_size[1] * input_filter,
                self.filters,
            )

            self.kernel = self.add_weight(
                shape=self.kernel_shape,
                initializer=self.kernel_initializer,
                name="kernel",
                regularizer=self.kernel_regularizer,
                constraint=self.kernel_constraint,
            )

        elif self.implementation == 2:
            if self.data_format == "channels_first":
                self.kernel_shape = (
                    input_filter,
                    input_row,
                    input_col,
                    self.filters,
                    self.output_row,
                    self.output_col,
                )
            else:
                self.kernel_shape = (
                    input_row,
                    input_col,
                    input_filter,
                    self.output_row,
                    self.output_col,
                    self.filters,
                )

            self.kernel = self.add_weight(
                shape=self.kernel_shape,
                initializer=self.kernel_initializer,
                name="kernel",
                regularizer=self.kernel_regularizer,
                constraint=self.kernel_constraint,
            )

            self.kernel_mask = (
                locally_connected_utils.get_locallyconnected_mask(
                    input_shape=(input_row, input_col),
                    kernel_shape=self.kernel_size,
                    strides=self.strides,
                    padding=self.padding,
                    data_format=self.data_format,
                )
            )

        elif self.implementation == 3:
            self.kernel_shape = (
                self.output_row * self.output_col * self.filters,
                input_row * input_col * input_filter,
            )

            self.kernel_idxs = sorted(
                conv_utils.conv_kernel_idxs(
                    input_shape=(input_row, input_col),
                    kernel_shape=self.kernel_size,
                    strides=self.strides,
                    padding=self.padding,
                    filters_in=input_filter,
                    filters_out=self.filters,
                    data_format=self.data_format,
                )
            )

            self.kernel = self.add_weight(
                shape=(len(self.kernel_idxs),),
                initializer=self.kernel_initializer,
                name="kernel",
                regularizer=self.kernel_regularizer,
                constraint=self.kernel_constraint,
            )

        else:
            raise ValueError(
                "Unrecognized implementation mode: %d." % self.implementation
            )

        if self.use_bias:
            self.bias = self.add_weight(
                shape=(output_row, output_col, self.filters),
                initializer=self.bias_initializer,
                name="bias",
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
            )
        else:
            self.bias = None
        if self.data_format == "channels_first":
            self.input_spec = InputSpec(ndim=4, axes={1: input_filter})
        else:
            self.input_spec = InputSpec(ndim=4, axes={-1: input_filter})
        self.built = True

    @tf_utils.shape_type_conversion
    def compute_output_shape(self, input_shape):
        if self.data_format == "channels_first":
            rows = input_shape[2]
            cols = input_shape[3]
        elif self.data_format == "channels_last":
            rows = input_shape[1]
            cols = input_shape[2]

        rows = conv_utils.conv_output_length(
            rows, self.kernel_size[0], self.padding, self.strides[0]
        )
        cols = conv_utils.conv_output_length(
            cols, self.kernel_size[1], self.padding, self.strides[1]
        )

        if self.data_format == "channels_first":
            return (input_shape[0], self.filters, rows, cols)
        elif self.data_format == "channels_last":
            return (input_shape[0], rows, cols, self.filters)

    def call(self, inputs):
        if self.implementation == 1:
            output = backend.local_conv(
                inputs,
                self.kernel,
                self.kernel_size,
                self.strides,
                (self.output_row, self.output_col),
                self.data_format,
            )

        elif self.implementation == 2:
            output = locally_connected_utils.local_conv_matmul(
                inputs,
                self.kernel,
                self.kernel_mask,
                self.compute_output_shape(inputs.shape),
            )

        elif self.implementation == 3:
            output = locally_connected_utils.local_conv_sparse_matmul(
                inputs,
                self.kernel,
                self.kernel_idxs,
                self.kernel_shape,
                self.compute_output_shape(inputs.shape),
            )

        else:
            raise ValueError(
                "Unrecognized implementation mode: %d." % self.implementation
            )

        if self.use_bias:
            output = backend.bias_add(
                output, self.bias, data_format=self.data_format
            )

        output = self.activation(output)
        return output

    def get_config(self):
        config = {
            "filters": self.filters,
            "kernel_size": self.kernel_size,
            "strides": self.strides,
            "padding": self.padding,
            "data_format": self.data_format,
            "activation": activations.serialize(self.activation),
            "use_bias": self.use_bias,
            "kernel_initializer": initializers.serialize(
                self.kernel_initializer
            ),
            "bias_initializer": initializers.serialize(self.bias_initializer),
            "kernel_regularizer": regularizers.serialize(
                self.kernel_regularizer
            ),
            "bias_regularizer": regularizers.serialize(self.bias_regularizer),
            "activity_regularizer": regularizers.serialize(
                self.activity_regularizer
            ),
            "kernel_constraint": constraints.serialize(self.kernel_constraint),
            "bias_constraint": constraints.serialize(self.bias_constraint),
            "implementation": self.implementation,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))

