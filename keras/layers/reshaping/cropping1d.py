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
"""Keras cropping layer for 1D input."""


import tensorflow.compat.v2 as tf

from keras.engine.base_layer import Layer
from keras.engine.input_spec import InputSpec
from keras.utils import conv_utils

# isort: off
from tensorflow.python.util.tf_export import keras_export


@keras_export("keras.layers.Cropping1D")
class Cropping1D(Layer):
    """Cropping layer for 1D input (e.g. temporal sequence).

    It crops along the time dimension (axis 1).

    Examples:

    >>> input_shape = (2, 3, 2)
    >>> x = np.arange(np.prod(input_shape)).reshape(input_shape)
    >>> print(x)
    [[[ 0  1]
      [ 2  3]
      [ 4  5]]
     [[ 6  7]
      [ 8  9]
      [10 11]]]
    >>> y = tf.keras.layers.Cropping1D(cropping=1)(x)
    >>> print(y)
    tf.Tensor(
      [[[2 3]]
       [[8 9]]], shape=(2, 1, 2), dtype=int64)

    Args:
      cropping: Int or tuple of int (length 2)
        How many units should be trimmed off at the beginning and end of
        the cropping dimension (axis 1).
        If a single int is provided, the same value will be used for both.

    Input shape:
      3D tensor with shape `(batch_size, axis_to_crop, features)`

    Output shape:
      3D tensor with shape `(batch_size, cropped_axis, features)`
    """

    def __init__(self, cropping=(1, 1), **kwargs):
        super().__init__(**kwargs)
        self.cropping = conv_utils.normalize_tuple(
            cropping, 2, "cropping", allow_zero=True
        )
        self.input_spec = InputSpec(ndim=3)

    def compute_output_shape(self, input_shape):
        input_shape = tf.TensorShape(input_shape).as_list()
        if input_shape[1] is not None:
            length = input_shape[1] - self.cropping[0] - self.cropping[1]
        else:
            length = None
        return tf.TensorShape([input_shape[0], length, input_shape[2]])

    def call(self, inputs):
        if (
            inputs.shape[1] is not None
            and sum(self.cropping) >= inputs.shape[1]
        ):
            raise ValueError(
                "cropping parameter of Cropping layer must be "
                "greater than the input shape. Received: inputs.shape="
                f"{inputs.shape}, and cropping={self.cropping}"
            )
        if self.cropping[1] == 0:
            return inputs[:, self.cropping[0] :, :]
        else:
            return inputs[:, self.cropping[0] : -self.cropping[1], :]

    def get_config(self):
        config = {"cropping": self.cropping}
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))
