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
"""Keras zero-padding layer for 1D input."""


import tensorflow.compat.v2 as tf

from keras import backend
from keras.engine.base_layer import Layer
from keras.engine.input_spec import InputSpec
from keras.utils import conv_utils

# isort: off
from tensorflow.python.util.tf_export import keras_export


@keras_export("keras.layers.ZeroPadding1D")
class ZeroPadding1D(Layer):
    """Zero-padding layer for 1D input (e.g. temporal sequence).

    Examples:

    >>> input_shape = (2, 2, 3)
    >>> x = np.arange(np.prod(input_shape)).reshape(input_shape)
    >>> print(x)
    [[[ 0  1  2]
      [ 3  4  5]]
     [[ 6  7  8]
      [ 9 10 11]]]
    >>> y = tf.keras.layers.ZeroPadding1D(padding=2)(x)
    >>> print(y)
    tf.Tensor(
      [[[ 0  0  0]
        [ 0  0  0]
        [ 0  1  2]
        [ 3  4  5]
        [ 0  0  0]
        [ 0  0  0]]
       [[ 0  0  0]
        [ 0  0  0]
        [ 6  7  8]
        [ 9 10 11]
        [ 0  0  0]
        [ 0  0  0]]], shape=(2, 6, 3), dtype=int64)

    Args:
        padding: Int, or tuple of int (length 2), or dictionary.
            - If int:
            How many zeros to add at the beginning and end of
            the padding dimension (axis 1).
            - If tuple of int (length 2):
            How many zeros to add at the beginning and the end of
            the padding dimension (`(left_pad, right_pad)`).

    Input shape:
        3D tensor with shape `(batch_size, axis_to_pad, features)`

    Output shape:
        3D tensor with shape `(batch_size, padded_axis, features)`
    """

    def __init__(self, padding=1, **kwargs):
        super().__init__(**kwargs)
        self.padding = conv_utils.normalize_tuple(
            padding, 2, "padding", allow_zero=True
        )
        self.input_spec = InputSpec(ndim=3)

    def compute_output_shape(self, input_shape):
        if input_shape[1] is not None:
            length = input_shape[1] + self.padding[0] + self.padding[1]
        else:
            length = None
        return tf.TensorShape([input_shape[0], length, input_shape[2]])

    def call(self, inputs):
        return backend.temporal_padding(inputs, padding=self.padding)

    def get_config(self):
        config = {"padding": self.padding}
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))
