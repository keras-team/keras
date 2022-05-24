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
"""Private base class for global pooling 1D layers."""
# pylint: disable=g-classes-have-attributes

from keras.engine.base_layer import Layer
from keras.engine.input_spec import InputSpec
from keras.utils import conv_utils
import tensorflow.compat.v2 as tf


class GlobalPooling1D(Layer):
    """Abstract class for different global pooling 1D layers."""

    def __init__(self, data_format="channels_last", keepdims=False, **kwargs):
        super().__init__(**kwargs)
        self.input_spec = InputSpec(ndim=3)
        self.data_format = conv_utils.normalize_data_format(data_format)
        self.keepdims = keepdims

    def compute_output_shape(self, input_shape):
        input_shape = tf.TensorShape(input_shape).as_list()
        if self.data_format == "channels_first":
            if self.keepdims:
                return tf.TensorShape([input_shape[0], input_shape[1], 1])
            else:
                return tf.TensorShape([input_shape[0], input_shape[1]])
        else:
            if self.keepdims:
                return tf.TensorShape([input_shape[0], 1, input_shape[2]])
            else:
                return tf.TensorShape([input_shape[0], input_shape[2]])

    def call(self, inputs):
        raise NotImplementedError

    def get_config(self):
        config = {"data_format": self.data_format, "keepdims": self.keepdims}
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))
