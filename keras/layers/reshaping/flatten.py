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
"""Contains the flatten layer."""
# pylint: disable=g-classes-have-attributes,g-direct-tensorflow-import

import functools
import operator

from keras.engine.base_layer import Layer
from keras.engine.input_spec import InputSpec
from keras.utils import conv_utils
import numpy as np
import tensorflow.compat.v2 as tf

from tensorflow.python.util.tf_export import keras_export


@keras_export("keras.layers.Flatten")
class Flatten(Layer):
    """Flattens the input. Does not affect the batch size.

    Note: If inputs are shaped `(batch,)` without a feature axis, then
    flattening adds an extra channel dimension and output shape is `(batch, 1)`.

    Args:
      data_format: A string,
        one of `channels_last` (default) or `channels_first`.
        The ordering of the dimensions in the inputs.
        `channels_last` corresponds to inputs with shape
        `(batch, ..., channels)` while `channels_first` corresponds to
        inputs with shape `(batch, channels, ...)`.
        It defaults to the `image_data_format` value found in your
        Keras config file at `~/.keras/keras.json`.
        If you never set it, then it will be "channels_last".

    Example:

    >>> model = tf.keras.Sequential()
    >>> model.add(tf.keras.layers.Conv2D(64, 3, 3, input_shape=(3, 32, 32)))
    >>> model.output_shape
    (None, 1, 10, 64)

    >>> model.add(Flatten())
    >>> model.output_shape
    (None, 640)

    """

    def __init__(self, data_format=None, **kwargs):
        super().__init__(**kwargs)
        self.data_format = conv_utils.normalize_data_format(data_format)
        self.input_spec = InputSpec(min_ndim=1)
        self._channels_first = self.data_format == "channels_first"

    def call(self, inputs):
        if self._channels_first:
            rank = inputs.shape.rank
            if rank and rank > 1:
                # Switch to channels-last format.
                permutation = [0]
                permutation.extend(range(2, rank))
                permutation.append(1)
                inputs = tf.transpose(inputs, perm=permutation)

        if tf.executing_eagerly():
            # Full static shape is guaranteed to be available.
            # Performance: Using `constant_op` is much faster than passing a list.
            flattened_shape = tf.constant([inputs.shape[0], -1])
            return tf.reshape(inputs, flattened_shape)
        else:
            input_shape = inputs.shape
            rank = input_shape.rank
            if rank == 1:
                return tf.expand_dims(inputs, axis=1)
            else:
                batch_dim = tf.compat.dimension_value(input_shape[0])
                non_batch_dims = input_shape[1:]
                # Reshape in a way that preserves as much shape info as possible.
                if non_batch_dims.is_fully_defined():
                    last_dim = int(
                        functools.reduce(operator.mul, non_batch_dims)
                    )
                    flattened_shape = tf.constant([-1, last_dim])
                elif batch_dim is not None:
                    flattened_shape = tf.constant([int(batch_dim), -1])
                else:
                    flattened_shape = [tf.shape(inputs)[0], -1]
                return tf.reshape(inputs, flattened_shape)

    def compute_output_shape(self, input_shape):
        input_shape = tf.TensorShape(input_shape).as_list()
        if not input_shape:
            output_shape = tf.TensorShape([1])
        else:
            output_shape = [input_shape[0]]
        if np.all(input_shape[1:]):
            output_shape += [np.prod(input_shape[1:], dtype=int)]
        else:
            output_shape += [None]
        return tf.TensorShape(output_shape)

    def get_config(self):
        config = super().get_config()
        config.update({"data_format": self.data_format})
        return config
